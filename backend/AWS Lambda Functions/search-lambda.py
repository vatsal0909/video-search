from fastapi import FastAPI, HTTPException
from mangum import Mangum
import json
import boto3
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict, Optional
from pydantic import BaseModel

app = FastAPI()


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
        
        print(f"Searching for: '{query_text}' (type: {search_type}, top_k: {top_k})")
        
        # Initialize clients
        opensearch_client = get_opensearch_client()
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Generate query embedding using Bedrock Marengo
        query_embedding = generate_text_embedding(bedrock_runtime, query_text)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        print(f"Generated embedding with {len(query_embedding)} dimensions")
        
        # Perform search based on type
        if search_type == 'hybrid':
            results = hybrid_search(opensearch_client, query_embedding, query_text, top_k)
        elif search_type == 'vector':
            results = vector_search(opensearch_client, query_embedding, top_k)
        elif search_type == 'text':
            results = text_search(opensearch_client, query_text, top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid search_type: {search_type}")
        
        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)
        
        print(f"Found {len(results)} results")
        
        return SearchResponse(
            query=query_text,
            search_type=search_type,
            total=len(results),
            clips=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in search: {str(e)}")
        import traceback
        traceback.print_exc()
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

            # print(video)
            
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
        print(f"Error in list_videos: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def get_opensearch_client():
    """Initialize OpenSearch Cluster client"""
    opensearch_host = os.environ['OPENSEARCH_CLUSTER_HOST']
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
        print(f"Error generating text embedding: {e}")
        return []


def convert_s3_to_presigned_urls(s3_client, results: List[Dict], expiration: int = 3600) -> List[Dict]:
    """Convert S3 paths to presigned URLs in video_path field"""
    for result in results:
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
                print(f"Error generating presigned URL for {video_path}: {e}")
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
        print(f"Error generating presigned URL for {video_path}: {e}")
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
        response = client.search(index="video_clips", body=search_body)
        
        videos = []
        for bucket in response['aggregations']['unique_videos']['buckets']:
            video_data = bucket['video_metadata']['hits']['hits'][0]['_source']
            video_data['clips_count'] = bucket['clip_count']['value']
            videos.append(video_data)
        
        return videos
        
    except Exception as e:
        print(f"Error fetching unique videos: {e}")
        return []


def hybrid_search(client, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
    """Hybrid search combining vector similarity and text matching"""
    search_body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": top_k # top_k * 2
                            }
                        }
                    },
                    {
                        "match": {
                            "clip_text": {
                                "query": query_text,
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ]
            }
        },
        "collapse": {
            "field": "clip_id"
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "embedding_scope"]
    }
    
    pipeline_exists = _create_hybrid_search_pipeline(client)
    
    if pipeline_exists:
        search_params = {
            "index": "video_clips",
            "body": search_body,
            "search_pipeline": "hybrid-norm-pipeline"
        }
    else:
        search_params = {
            "index": "video_clips",
            "body": search_body
        }
    
    try:
        response = client.search(**search_params)
        return parse_search_results(response)
        
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return vector_search(client, query_embedding, top_k)


def vector_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """Vector-only k-NN search"""
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        },
        "collapse": {
            "field": "clip_id"
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "embedding_scope"]
    }
    
    response = client.search(index="video_clips", body=search_body)
    return parse_search_results(response)


def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
    """Text-only BM25 search"""
    search_body = {
        "size": top_k,
        "query": {
            "match": {
                "clip_text": {
                    "query": query_text,
                    "fuzziness": "AUTO"
                }
            }
        },
        "collapse": {
            "field": "clip_id"
        },
        "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
                   "timestamp_end", "clip_text", "embedding_scope"]
    }
    
    response = client.search(index="video_clips", body=search_body)
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
                            "weights": [0.7, 0.3]
                        }
                    }
                }
            }
        ]
    }
    
    try:
        try:
            client.search_pipeline.get(id="hybrid-norm-pipeline")
            print("Hybrid search pipeline already exists")
        except:
            client.search_pipeline.put(
                id="hybrid-norm-pipeline",
                body=pipeline_body
            )
            print("✓ Created hybrid search pipeline with min-max normalization")
    
    except Exception as e:
        print(f"✗ Pipeline creation error: {e}")
        return False
    
    return True


def parse_search_results(response: Dict) -> List[Dict]:
    """Parse OpenSearch response into results list"""
    results = []
    
    for hit in response['hits']['hits']:
        result = hit['_source']
        result['score'] = hit['_score']
        result['_id'] = hit['_id']
        results.append(result)
    
    return results


lambda_handler = Mangum(app)


# import json
# import boto3
# import os
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# from typing import List, Dict


# def lambda_handler(event, context):
#     """
#     Perform hybrid search on OpenSearch Serverless
#     Combines text embedding + keyword matching
#     """
#     try:
#         # Extract search parameters
#         query_text = event.get('query_text', event.get('query', ''))
#         top_k = event.get('top_k', 10)
#         search_type = event.get('search_type', 'hybrid')  # 'hybrid', 'vector', 'text'
        
#         if not query_text:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': 'query_text is required'})
#             }
        
#         print(f"Searching for: '{query_text}' (type: {search_type}, top_k: {top_k})")
        
#         # Initialize clients
#         opensearch_client = get_opensearch_client()
#         bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
#         # Generate query embedding using Bedrock Marengo
#         query_embedding = generate_text_embedding(bedrock_runtime, query_text)
        
#         if not query_embedding:
#             return {
#                 'statusCode': 500,
#                 'body': json.dumps({'error': 'Failed to generate query embedding'})
#             }
        
#         print(f"Generated embedding with {len(query_embedding)} dimensions")
        
#         # Perform search based on type
#         if search_type == 'hybrid':
#             results = hybrid_search(opensearch_client, query_embedding, query_text, top_k)
#         elif search_type == 'vector':
#             results = vector_search(opensearch_client, query_embedding, top_k)
#         elif search_type == 'text':
#             results = text_search(opensearch_client, query_text, top_k)
#         else:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': f'Invalid search_type: {search_type}'})
#             }
        
#         print(f"Found {len(results)} results")
        
#         return {
#             'statusCode': 200,
#             'body': json.dumps({
#                 'query': query_text,
#                 'search_type': search_type,
#                 'total': len(results),
#                 'clips': results
#             }, default=str)
#         }
        
#     except Exception as e:
#         print(f"Error in hybrid search: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         return {
#             'statusCode': 500,
#             'body': json.dumps({'error': str(e)})
#         }


# def get_opensearch_client():
#     """Initialize OpenSearch Serverless client"""
#     opensearch_host = os.environ['OPENSEARCH_SERVERLESS_HOST']
#     opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
#     session = boto3.Session()
#     credentials = session.get_credentials()
    
#     awsauth = AWS4Auth(
#         credentials.access_key,
#         credentials.secret_key,
#         'us-east-1',
#         'aoss',
#         session_token=credentials.token
#     )
    
#     return OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=awsauth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20
#     )


# def generate_text_embedding(bedrock_runtime, text: str) -> List[float]:
#     """Generate embedding for text query using Bedrock Marengo"""
#     try:
#         request_body = {
#             "inputType": "text",
#             "inputText": text,
#             "textTruncate": "none"
#         }
        
#         response = bedrock_runtime.invoke_model(
#             modelId="us.twelvelabs.marengo-embed-2-7-v1:0",
#             body=json.dumps(request_body),
#             contentType="application/json",
#             accept="application/json"
#         )
        
#         result = json.loads(response['body'].read())
        
#         if 'data' in result and len(result['data']) > 0:
#             return result['data'][0].get('embedding', [])
        
#         return []
        
#     except Exception as e:
#         print(f"Error generating text embedding: {e}")
#         return []


# def hybrid_search(client, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
#     """
#     Hybrid search combining vector similarity and text matching
#     """
#     # search_body = {
#     #     "size": top_k,
#     #     "query": {
#     #         "hybrid": {
#     #             "queries": [
#     #                 {
#     #                     "knn": {
#     #                         "embedding": {
#     #                             "vector": query_embedding,
#     #                             "k": top_k * 2
#     #                         }
#     #                     }
#     #                 },
#     #                 {
#     #                     "match": {
#     #                         "clip_text": {
#     #                             "query": query_text,
#     #                             "fuzziness": "AUTO"
#     #                         }
#     #                     }
#     #                 }
#     #             ]
#     #         }
#     #     },
#     #     "_source": ["video_id", "video_path", "timestamp_start", 
#     #                "timestamp_end", "clip_text", "embedding_scope"]
#     # }

#     ## COLLAPSE Clause
#     # search_body = {
#     #     "size": top_k,
#     #     "query": {
#     #         "hybrid": {
#     #         "queries": [
#     #             {
#     #             "knn": {
#     #                 "embedding": {
#     #                 "vector": query_embedding,
#     #                 "k": top_k * 2
#     #                 }
#     #             }
#     #             },
#     #             {
#     #             "match": {
#     #                 "clip_text": {
#     #                 "query": query_text,
#     #                 "fuzziness": "AUTO"
#     #                 }
#     #             }
#     #             }
#     #         ]
#     #         }
#     #     },
#     #     "collapse": {
#     #         "field": "clip_id"
#     #     },
#     #     "_source": [
#     #         "video_id",
#     #         "video_path",
#     #         "clip_id",
#     #         "timestamp_start",
#     #         "timestamp_end",
#     #         "clip_text",
#     #         "embedding_scope"
#     #     ]
#     # }

#     # Filter IS NOT WORKING....
# #     search_body = {
# #     "size": top_k,
# #     "query": {
# #         "bool": {
# #             "must": {
# #                 "hybrid": {
# #                     "queries": [
# #                         {
# #                             "knn": {
# #                                 "embedding": {
# #                                     "vector": query_embedding,
# #                                     "k": top_k * 2
# #                                 }
# #                             }
# #                         },
# #                         {
# #                             "match": {
# #                                 "clip_text": {
# #                                     "query": query_text,
# #                                     "fuzziness": "AUTO"
# #                                 }
# #                             }
# #                         }
# #                     ]
# #                 }
# #             },
# #             "filter": [
# #                 { "term": { "embedding_scope": { "value": "visual-image" } } }
# #             ]
# #         }
# #     }
# # }

#     search_body = {
#         "size": top_k, ##### EXPLICITLY DONE, Top K are taken from below
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     {
#                         "knn": {
#                             "embedding": {
#                                 "vector": query_embedding,
#                                 "k": top_k * 2
#                             }
#                         }
#                     },
#                     {
#                         "match": {
#                             "clip_text": {
#                                 "query": query_text,
#                                 "fuzziness": "AUTO"
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "post_filter": {
#             "term": {
#                 "embedding_scope": "visual-image"
#             }
#         }
#     }

    
#     # Check if hybrid search pipeline exists
#     pipeline_exists = _create_hybrid_search_pipeline(client)
    
#     if pipeline_exists:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body,
#             "search_pipeline": "hybrid-norm-pipeline"
#         }
#     else:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body
#         }
    
#     try:
#         response = client.search(**search_params)
#         return parse_search_results(response)
        
#     except Exception as e:
#         print(f"Hybrid search error: {e}")
#         # Fallback to vector-only search
#         return vector_search(client, query_embedding, top_k)


# def vector_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
#     """Vector-only k-NN search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "knn": {
#                 "embedding": {
#                     "vector": query_embedding,
#                     "k": top_k
#                 }
#             }
#         },
#         "collapse": {
#             "field": "clip_id"
#         },
#         "_source": ["video_id", "video_path", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
#     """Text-only BM25 search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "match": {
#                 "clip_text": {
#                     "query": query_text,
#                     "fuzziness": "AUTO"
#                 }
#             }
#         },
#         "_source": ["video_id", "video_path", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def _create_hybrid_search_pipeline(client):
#         """Create search pipeline with score normalization for hybrid search"""
        
#         pipeline_body = {
#             "description": "Post-processing pipeline for hybrid search with normalization",
#             "phase_results_processors": [
#                 {
#                     "normalization-processor": {
#                         "normalization": {
#                             "technique": "min_max"
#                         },
#                         "combination": {
#                             "technique": "arithmetic_mean",
#                             "parameters": {
#                                 "weights": [0.7, 0.3]  # 70% vector, 30% text
#                             }
#                         }
#                     }
#                 }
#             ]
#         }
        
#         try:
#             # Check if pipeline exists
#             try:
#                 client.search_pipeline.get(id="hybrid-norm-pipeline")
#                 print("Hybrid search pipeline already exists")
#             except:
#                 # Create new pipeline
#                 client.search_pipeline.put(
#                     id="hybrid-norm-pipeline",
#                     body=pipeline_body
#                 )
#                 print("✓ Created hybrid search pipeline with min-max normalization")
        
#         except Exception as e:
#             print(f"✗ Pipeline creation error: {e}")
#             # Fallback: pipeline not supported, will use manual normalization
#             return False
        
#         return True


# def parse_search_results(response: Dict) -> List[Dict]:
#     """Parse OpenSearch response into results list"""
#     results = []
    
#     for hit in response['hits']['hits']:
#         result = hit['_source']
#         result['score'] = hit['_score']
#         result['_id'] = hit['_id']
#         results.append(result)
    
#     return results


################################# Serverless with presigned urls
# import json
# import boto3
# import os
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# from typing import List, Dict


# def lambda_handler(event, context):
#     """
#     Perform hybrid search on OpenSearch Serverless
#     Combines text embedding + keyword matching
#     """
#     try:
#         # Extract search parameters
#         query_text = event.get('query_text', event.get('query', ''))
#         top_k = event.get('top_k', 10)
#         search_type = event.get('search_type', 'hybrid')  # 'hybrid', 'vector', 'text'
        
#         if not query_text:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': 'query_text is required'})
#             }
        
#         print(f"Searching for: '{query_text}' (type: {search_type}, top_k: {top_k})")
        
#         # Initialize clients
#         opensearch_client = get_opensearch_client()
#         bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
#         s3_client = boto3.client('s3', region_name='us-east-1')
        
#         # Generate query embedding using Bedrock Marengo
#         query_embedding = generate_text_embedding(bedrock_runtime, query_text)
        
#         if not query_embedding:
#             return {
#                 'statusCode': 500,
#                 'body': json.dumps({'error': 'Failed to generate query embedding'})
#             }
        
#         print(f"Generated embedding with {len(query_embedding)} dimensions")
        
#         # Perform search based on type
#         if search_type == 'hybrid':
#             results = hybrid_search(opensearch_client, query_embedding, query_text, top_k)
#         elif search_type == 'vector':
#             results = vector_search(opensearch_client, query_embedding, top_k)
#         elif search_type == 'text':
#             results = text_search(opensearch_client, query_text, top_k)
#         else:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': f'Invalid search_type: {search_type}'})
#             }
        
#         # Convert S3 paths to presigned URLs
#         results = convert_s3_to_presigned_urls(s3_client, results)
        
#         print(f"Found {len(results)} results")
        
#         return {
#             'statusCode': 200,
#             'body': json.dumps({
#                 'query': query_text,
#                 'search_type': search_type,
#                 'total': len(results),
#                 'clips': results
#             }, default=str)
#         }
        
#     except Exception as e:
#         print(f"Error in hybrid search: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         return {
#             'statusCode': 500,
#             'body': json.dumps({'error': str(e)})
#         }


# def get_opensearch_client():
#     """Initialize OpenSearch Serverless client"""
#     opensearch_host = os.environ['OPENSEARCH_SERVERLESS_HOST']
#     opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
#     session = boto3.Session()
#     credentials = session.get_credentials()
    
#     awsauth = AWS4Auth(
#         credentials.access_key,
#         credentials.secret_key,
#         'us-east-1',
#         'aoss',
#         session_token=credentials.token
#     )
    
#     return OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=awsauth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20
#     )


# def generate_text_embedding(bedrock_runtime, text: str) -> List[float]:
#     """Generate embedding for text query using Bedrock Marengo"""
#     try:
#         request_body = {
#             "inputType": "text",
#             "inputText": text,
#             "textTruncate": "none"
#         }
        
#         response = bedrock_runtime.invoke_model(
#             modelId="us.twelvelabs.marengo-embed-2-7-v1:0",
#             body=json.dumps(request_body),
#             contentType="application/json",
#             accept="application/json"
#         )
        
#         result = json.loads(response['body'].read())
        
#         if 'data' in result and len(result['data']) > 0:
#             return result['data'][0].get('embedding', [])
        
#         return []
        
#     except Exception as e:
#         print(f"Error generating text embedding: {e}")
#         return []


# def convert_s3_to_presigned_urls(s3_client, results: List[Dict], expiration: int = 3600) -> List[Dict]:
#     """
#     Convert S3 paths to presigned URLs in video_path field
    
#     Args:
#         s3_client: Boto3 S3 client
#         results: List of search results
#         expiration: URL expiration time in seconds (default: 1 hour)
    
#     Returns:
#         Results with presigned URLs in video_path
#     """
#     for result in results:
#         video_path = result.get('video_path', '')
        
#         if video_path.startswith('s3://'):
#             try:
#                 # Parse S3 URI
#                 s3_parts = video_path.replace('s3://', '').split('/', 1)
#                 bucket = s3_parts[0]
#                 key = s3_parts[1] if len(s3_parts) > 1 else ''
                
#                 # Generate presigned URL
#                 presigned_url = s3_client.generate_presigned_url(
#                     'get_object',
#                     Params={'Bucket': bucket, 'Key': key},
#                     ExpiresIn=expiration
#                 )
                
#                 # Replace S3 path with presigned URL
#                 result['video_path'] = presigned_url
                
#             except Exception as e:
#                 print(f"Error generating presigned URL for {video_path}: {e}")
#                 # Keep original S3 path if error
#                 pass
    
#     return results


# def hybrid_search(client, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
#     """
#     Hybrid search combining vector similarity and text matching
#     """
#     search_body = {
#         "size": top_k,
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     {
#                         "knn": {
#                             "embedding": {
#                                 "vector": query_embedding,
#                                 "k": top_k * 2
#                             }
#                         }
#                     },
#                     {
#                         "match": {
#                             "clip_text": {
#                                 "query": query_text,
#                                 "fuzziness": "AUTO"
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "post_filter": {
#             "term": {
#                 "embedding_scope": "visual-image"
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     # Check if hybrid search pipeline exists
#     pipeline_exists = _create_hybrid_search_pipeline(client)
    
#     if pipeline_exists:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body,
#             "search_pipeline": "hybrid-norm-pipeline"
#         }
#     else:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body
#         }
    
#     try:
#         response = client.search(**search_params)
#         return parse_search_results(response)
        
#     except Exception as e:
#         print(f"Hybrid search error: {e}")
#         # Fallback to vector-only search
#         return vector_search(client, query_embedding, top_k)


# def vector_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
#     """Vector-only k-NN search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "knn": {
#                 "embedding": {
#                     "vector": query_embedding,
#                     "k": top_k
#                 }
#             }
#         },
#         "collapse": {
#             "field": "clip_id"
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
#     """Text-only BM25 search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "match": {
#                 "clip_text": {
#                     "query": query_text,
#                     "fuzziness": "AUTO"
#                 }
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def _create_hybrid_search_pipeline(client):
#     """Create search pipeline with score normalization for hybrid search"""
    
#     pipeline_body = {
#         "description": "Post-processing pipeline for hybrid search with normalization",
#         "phase_results_processors": [
#             {
#                 "normalization-processor": {
#                     "normalization": {
#                         "technique": "min_max"
#                     },
#                     "combination": {
#                         "technique": "arithmetic_mean",
#                         "parameters": {
#                             "weights": [0.7, 0.3]  # 70% vector, 30% text
#                         }
#                     }
#                 }
#             }
#         ]
#     }
    
#     try:
#         # Check if pipeline exists
#         try:
#             client.search_pipeline.get(id="hybrid-norm-pipeline")
#             print("Hybrid search pipeline already exists")
#         except:
#             # Create new pipeline
#             client.search_pipeline.put(
#                 id="hybrid-norm-pipeline",
#                 body=pipeline_body
#             )
#             print("✓ Created hybrid search pipeline with min-max normalization")
    
#     except Exception as e:
#         print(f"✗ Pipeline creation error: {e}")
#         # Fallback: pipeline not supported, will use manual normalization
#         return False
    
#     return True


# def parse_search_results(response: Dict) -> List[Dict]:
#     """Parse OpenSearch response into results list"""
#     results = []
    
#     for hit in response['hits']['hits']:
#         result = hit['_source']
#         result['score'] = hit['_score']
#         result['_id'] = hit['_id']
#         results.append(result)
    
#     return results

############################ Cluster with presigned urls


# import json
# import boto3
# import os
# from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
# from typing import List, Dict


# def lambda_handler(event, context):
#     """
#     Perform hybrid search on OpenSearch Cluster
#     Combines text embedding + keyword matching
#     """
#     try:
#         # Extract search parameters
#         query_text = event.get('query_text', event.get('query', ''))
#         top_k = event.get('top_k', 10)
#         search_type = event.get('search_type', 'hybrid')  # 'hybrid', 'vector', 'text'
        
#         if not query_text:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': 'query_text is required'})
#             }
        
#         print(f"Searching for: '{query_text}' (type: {search_type}, top_k: {top_k})")
        
#         # Initialize clients
#         opensearch_client = get_opensearch_client()
#         bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
#         s3_client = boto3.client('s3', region_name='us-east-1')
        
#         # Generate query embedding using Bedrock Marengo
#         query_embedding = generate_text_embedding(bedrock_runtime, query_text)
        
#         if not query_embedding:
#             return {
#                 'statusCode': 500,
#                 'body': json.dumps({'error': 'Failed to generate query embedding'})
#             }
        
#         print(f"Generated embedding with {len(query_embedding)} dimensions")
        
#         # Perform search based on type
#         if search_type == 'hybrid':
#             results = hybrid_search(opensearch_client, query_embedding, query_text, top_k)
#         elif search_type == 'vector':
#             results = vector_search(opensearch_client, query_embedding, top_k)
#         elif search_type == 'text':
#             results = text_search(opensearch_client, query_text, top_k)
#         else:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps({'error': f'Invalid search_type: {search_type}'})
#             }
        
#         # Convert S3 paths to presigned URLs
#         results = convert_s3_to_presigned_urls(s3_client, results)
        
#         print(f"Found {len(results)} results")
        
#         return {
#             'statusCode': 200,
#             'body': json.dumps({
#                 'query': query_text,
#                 'search_type': search_type,
#                 'total': len(results),
#                 'clips': results
#             }, default=str)
#         }
        
#     except Exception as e:
#         print(f"Error in hybrid search: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         return {
#             'statusCode': 500,
#             'body': json.dumps({'error': str(e)})
#         }


# def get_opensearch_client():
#     """Initialize OpenSearch Cluster client (CHANGED from Serverless)"""
#     # CHANGE 1: Use cluster endpoint environment variable
#     opensearch_host = os.environ['OPENSEARCH_CLUSTER_HOST']  # e.g., search-demo-cluster-xxx.us-east-1.es.amazonaws.com
#     opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
#     # CHANGE 2: Use AWSV4SignerAuth instead of AWS4Auth
#     session = boto3.Session()
#     credentials = session.get_credentials()
    
#     auth = AWSV4SignerAuth(credentials, 'us-east-1', 'es')  # 'es' instead of 'aoss'
    
#     # CHANGE 3: Use new auth object
#     return OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=auth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20
#     )


# def generate_text_embedding(bedrock_runtime, text: str) -> List[float]:
#     """Generate embedding for text query using Bedrock Marengo"""
#     try:
#         request_body = {
#             "inputType": "text",
#             "inputText": text,
#             "textTruncate": "none"
#         }
        
#         response = bedrock_runtime.invoke_model(
#             modelId="us.twelvelabs.marengo-embed-2-7-v1:0",
#             body=json.dumps(request_body),
#             contentType="application/json",
#             accept="application/json"
#         )
        
#         result = json.loads(response['body'].read())
        
#         if 'data' in result and len(result['data']) > 0:
#             return result['data'][0].get('embedding', [])
        
#         return []
        
#     except Exception as e:
#         print(f"Error generating text embedding: {e}")
#         return []


# def convert_s3_to_presigned_urls(s3_client, results: List[Dict], expiration: int = 3600) -> List[Dict]:
#     """Convert S3 paths to presigned URLs in video_path field"""
#     for result in results:
#         video_path = result.get('video_path', '')
        
#         if video_path.startswith('s3://'):
#             try:
#                 # Parse S3 URI
#                 s3_parts = video_path.replace('s3://', '').split('/', 1)
#                 bucket = s3_parts[0]
#                 key = s3_parts[1] if len(s3_parts) > 1 else ''
                
#                 # Generate presigned URL
#                 presigned_url = s3_client.generate_presigned_url(
#                     'get_object',
#                     Params={'Bucket': bucket, 'Key': key},
#                     ExpiresIn=expiration
#                 )
                
#                 # Replace S3 path with presigned URL
#                 result['video_path'] = presigned_url
                
#             except Exception as e:
#                 print(f"Error generating presigned URL for {video_path}: {e}")
#                 pass
    
#     return results


# def hybrid_search(client, query_embedding: List[float], query_text: str, top_k: int = 10) -> List[Dict]:
#     """Hybrid search combining vector similarity and text matching"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     {
#                         "knn": {
#                             "embedding": {
#                                 "vector": query_embedding,
#                                 "k": top_k * 2
#                             }
#                         }
#                     },
#                     {
#                         "match": {
#                             "clip_text": {
#                                 "query": query_text,
#                                 "fuzziness": "AUTO"
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "collapse":{
#             "field": "clip_id"
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     # CHANGE 4: Cluster supports search pipelines (unlike Serverless)
#     pipeline_exists = _create_hybrid_search_pipeline(client)
    
#     if pipeline_exists:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body,
#             "search_pipeline": "hybrid-norm-pipeline"
#         }
#     else:
#         search_params = {
#             "index": "video_clips",
#             "body": search_body
#         }
    
#     try:
#         response = client.search(**search_params)
#         return parse_search_results(response)
        
#     except Exception as e:
#         print(f"Hybrid search error: {e}")
#         return vector_search(client, query_embedding, top_k)


# def vector_search(client, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
#     """Vector-only k-NN search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "knn": {
#                 "embedding": {
#                     "vector": query_embedding,
#                     "k": top_k
#                 }
#             }
#         },
#         "collapse": {
#             "field": "clip_id"
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
#     """Text-only BM25 search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "match": {
#                 "clip_text": {
#                     "query": query_text,
#                     "fuzziness": "AUTO"
#                 }
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start", 
#                    "timestamp_end", "clip_text", "embedding_scope"]
#     }
    
#     response = client.search(index="video_clips", body=search_body)
#     return parse_search_results(response)


# def _create_hybrid_search_pipeline(client):
#     """Create search pipeline with score normalization for hybrid search"""
    
#     pipeline_body = {
#         "description": "Post-processing pipeline for hybrid search with normalization",
#         "phase_results_processors": [
#             {
#                 "normalization-processor": {
#                     "normalization": {
#                         "technique": "min_max"
#                     },
#                     "combination": {
#                         "technique": "arithmetic_mean",
#                         "parameters": {
#                             "weights": [0.7, 0.3]  # 70% vector, 30% text
#                         }
#                     }
#                 }
#             }
#         ]
#     }
    
#     try:
#         # Check if pipeline exists
#         try:
#             client.search_pipeline.get(id="hybrid-norm-pipeline")
#             print("Hybrid search pipeline already exists")
#         except:
#             # Create new pipeline
#             client.search_pipeline.put(
#                 id="hybrid-norm-pipeline",
#                 body=pipeline_body
#             )
#             print("✓ Created hybrid search pipeline with min-max normalization")
    
#     except Exception as e:
#         print(f"✗ Pipeline creation error: {e}")
#         return False
    
#     return True


# def parse_search_results(response: Dict) -> List[Dict]:
#     """Parse OpenSearch response into results list"""
#     results = []
    
#     for hit in response['hits']['hits']:
#         result = hit['_source']
#         result['score'] = hit['_score']
#         result['_id'] = hit['_id']
#         results.append(result)
    
#     return results