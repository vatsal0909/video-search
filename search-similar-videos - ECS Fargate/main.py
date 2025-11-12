from fastapi import FastAPI, HTTPException
import json
import boto3
import os
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict, Optional
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Video Search Service", version="1.0.0")


# CHANGE 1: Updated index name to consolidated index
INDEX_NAME = "video_clips_consolidated"
VECTOR_PIPELINE = "vector-norm-pipeline-consolidated-index-rrf"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://condenast-fe.s3-website-us-east-1.amazonaws.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 10
    search_type: str = "hybrid"


class VideoMetadata(BaseModel):
    video_id: str
    video_path: str
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None
    upload_date: Optional[str] = None
    clips_count: int = 0


class VideosListResponse(BaseModel):
    videos: List[VideoMetadata]
    total: int


class SearchResponse(BaseModel):
    query: str
    search_type: str
    total: int
    clips: List[Dict]


@app.get("/health")
async def health_check():
    """Health check endpoint for ECS task"""
    return {"status": "healthy", "service": "video-search"}


@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Search videos using hybrid/vector/text search
    Performs hybrid search on OpenSearch Cluster
    Combines text embedding + keyword matching
    """
    try:
        query_text = request.query_text
        top_k = request.top_k
        search_type = request.search_type
        
        if not query_text:
            raise HTTPException(status_code=400, detail="query_text is required")
        
        logger.info(f"Searching for: '{query_text}' (type: {search_type}, top_k: {top_k})")
        
        # Initialize clients
        opensearch_client = get_opensearch_client()
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Generate query embedding using Bedrock Marengo
        query_embedding = generate_text_embedding(bedrock_runtime, query_text)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        logger.info(f"Generated embedding with {len(query_embedding)} dimensions")
        
        # Perform search based on type
        if search_type == 'hybrid':
            results = hybrid_search(opensearch_client, query_embedding, query_text, top_k)
        elif search_type == 'vector':
            results = vector_search(opensearch_client, query_embedding, top_k)
        elif search_type == 'visual':
            results = visual_search(opensearch_client, query_embedding, top_k)
        elif search_type == 'audio':
            results = audio_search(opensearch_client, query_embedding, top_k)
        elif search_type == 'text':
            results = text_search(opensearch_client, query_text, top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid search_type: {search_type}")
        
        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)
        
        logger.info(f"Found {len(results)} results")
        
        return SearchResponse(
            query=query_text,
            search_type=search_type,
            total=len(results),
            clips=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=VideosListResponse)
async def list_all_videos():
    """
    Get all unique videos from the OpenSearch index
    Returns video metadata including S3 paths and clip counts
    """
    try:
        opensearch_client = get_opensearch_client()
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Get all unique videos from OpenSearch
        videos = get_all_unique_videos(opensearch_client)
        
        # Transform to response format
        video_list = []
        for video in videos:
            # Generate presigned URL for private S3 bucket access
            presigned_url = convert_s3_to_presigned_url(s3_client, video['video_path'])

            video_list.append(VideoMetadata(
                video_id=video['video_id'],
                video_path=presigned_url if presigned_url else video['video_path'],
                title=video.get('clip_text') or f"Video {video['video_id'][:8]}",
                thumbnail_url=video.get('thumbnail_url'),
                duration=video.get('duration'),
                upload_date=video.get('upload_date'),
                clips_count=video.get('clips_count', 0)
            ))
        
        return VideosListResponse(
            videos=video_list,
            total=len(video_list)
        )
    
    except Exception as e:
        logger.error(f"Error in list_videos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_opensearch_client():
    """Initialize OpenSearch Cluster client"""
    opensearch_host = os.environ.get('OPENSEARCH_CLUSTER_HOST')
    if not opensearch_host:
        raise ValueError("OPENSEARCH_CLUSTER_HOST environment variable not set")
    
    opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
    session = boto3.Session()
    credentials = session.get_credentials()
    
    auth = AWSV4SignerAuth(credentials, 'us-east-1', 'es')
    
    return OpenSearch(
        hosts=[{'host': opensearch_host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )


def generate_text_embedding(bedrock_runtime, text: str) -> List[float]:
    """Generate embedding for text query using Bedrock Marengo"""
    try:
        request_body = {
            "inputType": "text",
            "inputText": text,
            "textTruncate": "none"
        }
        
        response = bedrock_runtime.invoke_model(
            modelId="us.twelvelabs.marengo-embed-2-7-v1:0", 
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response['body'].read())
        
        if 'data' in result and len(result['data']) > 0:
            return result['data'][0].get('embedding', [])
        
        return []
        
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}", exc_info=True)
        return []


def convert_s3_to_presigned_urls(s3_client, results: List[Dict], expiration: int = 3600) -> List[Dict]:
    """Convert S3 paths to presigned URLs in video_path and thumbnail_path fields"""
    for result in results:
        # Convert video_path to presigned URL
        video_path = result.get('video_path', '')
        if video_path.startswith('s3://'):
            try:
                s3_parts = video_path.replace('s3://', '').split('/', 1)
                bucket = s3_parts[0]
                key = s3_parts[1] if len(s3_parts) > 1 else ''
                
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expiration
                )
                
                result['video_path'] = presigned_url
                
            except Exception as e:
                logger.warning(f"Error generating presigned URL for {video_path}: {e}")
                pass
        
        # Convert thumbnail_path to presigned URL
        thumbnail_path = result.get('thumbnail_path', '')
        if thumbnail_path and thumbnail_path.startswith('s3://'):
            try:
                s3_parts = thumbnail_path.replace('s3://', '').split('/', 1)
                bucket = s3_parts[0]
                key = s3_parts[1] if len(s3_parts) > 1 else ''
                
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expiration
                )
                
                result['thumbnail_path'] = presigned_url
                # logger.info(f"✓ Generated presigned URL for thumbnail: {key}")
                
            except Exception as e:
                logger.warning(f"Error generating presigned URL for thumbnail {thumbnail_path}: {e}")
                pass
    
    return results


def convert_s3_to_presigned_url(s3_client, video_path: str, expiration: int = 3600) -> Optional[str]:
    """Convert single S3 path to presigned URL"""
    if not video_path.startswith('s3://'):
        return None
    
    try:
        s3_parts = video_path.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1] if len(s3_parts) > 1 else ''
        
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
        return presigned_url
        
    except Exception as e:
        logger.warning(f"Error generating presigned URL for {video_path}: {e}")
        return None


def get_all_unique_videos(client) -> List[Dict]:
    """Get all unique videos from OpenSearch index"""
    search_body = {
        "size": 0,
        "aggs": {
            "unique_videos": {
                "terms": {
                    "field": "video_id",
                    "size": 10000
                },
                "aggs": {
                    "video_metadata": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["video_id", "video_path", "clip_text"]
                        }
                    },
                    "clip_count": {
                        "cardinality": {
                            "field": "clip_id"
                        }
                    }
                }
            }
        }
    }
    
    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        
        videos = []
        for bucket in response['aggregations']['unique_videos']['buckets']:
            video_data = bucket['video_metadata']['hits']['hits'][0]['_source']
            video_data['clips_count'] = bucket['clip_count']['value']
            videos.append(video_data)
        
        return videos
        
    except Exception as e:
        logger.error(f"Error fetching unique videos: {e}", exc_info=True)
        return []


# CHANGE 3: Updated hybrid_search to query emb_vis_text and emb_audio
def hybrid_search(client, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
    """Hybrid search combining vector similarity on visual-text & audio + text matching"""
    search_body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    # Visual-text embedding (k-NN) - weight 0.5
                    {
                        "knn": {
                            "emb_vis_text": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    },
                    # Audio embedding (k-NN) - weight 0.3
                    {
                        "knn": {
                            "emb_audio": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    },
                    # Text matching (BM25) - weight 0.2
                    {
                        "match": {
                            "video_name": {
                                "query": query_text,
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration"]
    }
    pipeline_exists = _create_hybrid_search_pipeline(client)
    if pipeline_exists:
        search_params = {
                "index": INDEX_NAME,
                "body": search_body,
                "search_pipeline": "hybrid-norm-pipeline"
            }
    else:
        search_params = {
                "index": INDEX_NAME,
                "body": search_body
            }
    
    try:
        response = client.search(**search_params)
        return parse_search_results(response)
        
    except Exception as e:
        logger.error(f"Hybrid search error: {e}", exc_info=True)
        return vector_search(client, query_embedding, top_k)


# CHANGE 4: Updated vector_search to query emb_vis_text and emb_audio
def vector_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """Vector-only k-NN search on visual-text and audio embeddings with normalization"""
    search_body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    # ordering here should match weights defined in the pipeline (0.7, 0.3)
                    {
                        "knn": {
                            "emb_vis_text": 
                            {
                                "vector": query_embedding, 
                                "k": top_k
                            }
                        }
                    },
                    {
                        "knn": {
                            "emb_audio": 
                            {
                                "vector": query_embedding, 
                                "k": top_k
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start",
                    "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration"]
    }
    pipeline_exists = _create_vector_search_pipeline(client)
    if pipeline_exists:
        search_params = {
                "index": INDEX_NAME,
                "body": search_body,
                "search_pipeline": VECTOR_PIPELINE
            }
    else:
        search_params = {
                "index": INDEX_NAME,
                "body": search_body
            }

    response = client.search(**search_params)
    return parse_search_results(response)


def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
    """Text-only BM25 search"""
    search_body = {
        "size": top_k,
        "query": {
            "match": {
                "video_name": {
                    "query": query_text,
                    "fuzziness": "AUTO"
                }
            }
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text",  "thumbnail_path", "video_name", "clip_duration"]
    }
    
    response = client.search(index=INDEX_NAME, body=search_body)
    return parse_search_results(response)


def visual_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """Visual-only k-NN search on visual-text embeddings"""
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "emb_vis_text": {
                    "vector": query_embedding,
                    "min_score": 0.6
                }
            }
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration"]
    }
    
    response = client.search(index=INDEX_NAME, body=search_body)
    return parse_search_results(response)


def audio_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """Audio-only k-NN search on audio embeddings"""
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "emb_audio": {
                    "vector": query_embedding,
                    "min_score": 0.6
                }
            }
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration"]
    }
    
    response = client.search(index=INDEX_NAME, body=search_body)
    return parse_search_results(response)


def _create_hybrid_search_pipeline(client):
    """Create search pipeline with score normalization for hybrid search"""
    
    pipeline_body = {
        "description": "Post-processing pipeline for hybrid search with normalization",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [0.5, 0.3, 0.2]
                        }
                    }
                }
            }
        ]
    }
    
    try:
        try:
            client.search_pipeline.get(id="hybrid-norm-pipeline-consolidated-index")
            logger.info("Hybrid search pipeline already exists")
        except:
            client.search_pipeline.put(
                id="hybrid-norm-pipeline-consolidated-index",
                body=pipeline_body
            )
            logger.info("✓ Created hybrid search pipeline with min-max normalization")
    
    except Exception as e:
        logger.warning(f"✗ Pipeline creation error: {e}")
        return False
    
    return True


def _create_vector_search_pipeline(client):
    """Create search pipeline with score normalization for vector search"""
    
    # pipeline_body = {
    #     "description": "Post-processing pipeline for vector search with min-max normalization (0-1 range)",
    #     "phase_results_processors": [
    #         {
    #             "normalization-processor": {
    #                 "normalization": {
    #                     "technique": "min_max"
    #                 },
    #                 "combination": {
    #                     "technique": "arithmetic_mean",
    #                     "parameters": {
    #                         "weights": [0.6, 0.4]
    #                     }
    #                 }
    #             }
    #         }
    #     ]
    # }
    pipeline_body = {
        "description": "Post processor for hybrid RRF search",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 1
                    }
                }
            }
        ]
    }
    
    try:
        try:
            client.search_pipeline.get(id=VECTOR_PIPELINE)
            logger.info("✓ Vector search pipeline already exists")
        except:
            client.search_pipeline.put(
                id=VECTOR_PIPELINE,
                body=pipeline_body
            )
            logger.info("✓ Created vector search pipeline with min-max normalization")
    
    except Exception as e:
        logger.warning(f"✗ Vector pipeline creation error: {e}")
        return False
    
    return True


def parse_search_results(response: Dict) -> List[Dict]:
    """Parse OpenSearch response into results list"""
    results = []
    
    for hit in response['hits']['hits']:
        result = hit['_source']
        result['score'] = hit['_score']
        result['_id'] = hit['_id']
        # logger.info(result)
        results.append(result)
    
    return results


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
