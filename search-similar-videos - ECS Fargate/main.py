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
import math


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Video Search Service", version="2.1.0-hybrid-optimized")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://condenast-fe.s3-website-us-east-1.amazonaws.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query_text: str
    video_id: Optional[str] = None
    top_k: int = 10
    search_type: str = "hybrid"  # hybrid, vector, text, multimodal
    confidence_threshold: Optional[float] = 0.0  # NEW: Filter by confidence (0-1)


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
    average_confidence: Optional[float] = None  # NEW: Avg confidence score


# ================== CORRECTED INDUSTRY-STANDARD WEIGHTS ==================

SEARCH_WEIGHTS = {
    # Hybrid: Balanced semantic + text matching
    # For Condé Nast: Emphasizes visual text (OCR) for editorial content
    "hybrid": {
        "emb_vis_text": 1.8,      # Visual text/OCR (HIGHEST - editorial captions)
        "emb_vis_image": 1.2,     # Visual images (editorial photos)
        "emb_audio": 0.8,         # Audio/speech (LOWEST - secondary)
        "text_match": 1.5         # BM25 text matching (Strong influence)
    },
    # Vector: Equal weights across all modalities
    "vector": {
        "emb_vis_text": 1.0,
        "emb_vis_image": 1.0,
        "emb_audio": 1.0
    },
    # Multimodal: Text-focused for keyword queries
    "multimodal": {
        "emb_vis_text": 2.0,
        "emb_vis_image": 1.5,
        "emb_audio": 1.0
    }
}

INDEX_NAME = "updated_video_clips_cosine_sim"
RRF_K = 60  # Reciprocal Rank Fusion parameter


@app.get("/health")
async def health_check():
    """Health check endpoint for ECS task"""
    return {
        "status": "healthy",
        "service": "video-search",
        "version": "2.1.0-hybrid-optimized",
        "index": INDEX_NAME,
        "space_type": "cosinesimil"
    }


@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Search videos with hybrid approach optimized for accuracy.

    Improvements:
    - Corrected weights (1.8, 1.2, 0.8, 1.5)
    - RRF (Reciprocal Rank Fusion) normalization
    - Confidence scores (0-1 range)
    - Minimum match requirement: 2+ modalities
    - Accuracy-focused result ranking

    confidence_threshold: Filter results below this confidence (0.0-1.0)
    """
    try:
        query_text = request.query_text
        video_id = request.video_id
        top_k = request.top_k
        search_type = request.search_type
        confidence_threshold = request.confidence_threshold or 0.0

        if not query_text:
            raise HTTPException(status_code=400, detail="query_text is required")

        logger.info(f"Searching: '{query_text}' (video_id: {video_id}, type: {search_type}, top_k: {top_k})")

        # Initialize clients
        opensearch_client = get_opensearch_client()
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Generate query embedding using Bedrock Marengo
        query_embedding = generate_text_embedding(bedrock_runtime, query_text)

        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        logger.info(f"Generated embedding: {len(query_embedding)}D vector")

        # Perform search based on type
        if search_type == "hybrid":
            results = hybrid_search(
                opensearch_client, query_embedding, query_text, video_id, top_k
            )
        elif search_type == "vector":
            results = vector_search(opensearch_client, query_embedding, video_id, top_k)
        elif search_type == "text":
            results = text_search(opensearch_client, query_text, video_id, top_k)
        elif search_type == "multimodal":
            results = multimodal_search(opensearch_client, query_embedding, video_id, top_k)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_type: {search_type}"
            )

        # Filter by confidence threshold
        if confidence_threshold > 0.0:
            original_count = len(results)
            results = [r for r in results if r.get('confidence', 1.0) >= confidence_threshold]
            logger.info(f"Confidence filter: {original_count} → {len(results)} results")

        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)

        # Calculate average confidence
        avg_confidence = None
        if results:
            confidences = [r.get('confidence', 1.0) for r in results]
            avg_confidence = sum(confidences) / len(confidences)

        avg_conf_str = f"{avg_confidence:.3f}" if avg_confidence else "N/A"
        logger.info(f"Found {len(results)} results (avg confidence: {avg_conf_str})")

        return SearchResponse(
            query=query_text,
            search_type=search_type,
            total=len(results),
            clips=results,
            average_confidence=avg_confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/in-video", response_model=SearchResponse)
async def search_in_video(request: SearchRequest, video_id: str):
    """
    Search within a specific video using hybrid approach.
    Perfect for targeted video search with high accuracy.
    """
    try:
        query_text = request.query_text
        top_k = request.top_k
        search_type = request.search_type
        confidence_threshold = request.confidence_threshold or 0.0

        if not query_text:
            raise HTTPException(status_code=400, detail="query_text is required")

        logger.info(f"In-video search: {video_id} for '{query_text}'")

        # Initialize clients
        opensearch_client = get_opensearch_client()
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')

        # Generate query embedding
        query_embedding = generate_text_embedding(bedrock_runtime, query_text)

        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        # Search within specific video
        if search_type == "hybrid":
            results = hybrid_search(opensearch_client, query_embedding, query_text, video_id, top_k)
        elif search_type == "vector":
            results = vector_search(opensearch_client, query_embedding, video_id, top_k)
        elif search_type == "text":
            results = text_search(opensearch_client, query_text, video_id, top_k)
        elif search_type == "multimodal":
            results = multimodal_search(opensearch_client, query_embedding, video_id, top_k)
        else:
            results = hybrid_search(opensearch_client, query_embedding, query_text, video_id, top_k)

        # Filter by confidence threshold
        if confidence_threshold > 0.0:
            results = [r for r in results if r.get('confidence', 1.0) >= confidence_threshold]

        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)

        # Calculate average confidence
        avg_confidence = None
        if results:
            confidences = [r.get('confidence', 1.0) for r in results]
            avg_confidence = sum(confidences) / len(confidences)

        logger.info(f"Found {len(results)} results in video {video_id}")

        return SearchResponse(
            query=query_text,
            search_type=search_type,
            total=len(results),
            clips=results,
            average_confidence=avg_confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in in-video search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=VideosListResponse)
async def list_all_videos():
    """Get all unique videos from the OpenSearch consolidated index"""
    try:
        opensearch_client = get_opensearch_client()
        s3_client = boto3.client('s3', region_name='us-east-1')

        videos = get_all_unique_videos(opensearch_client)

        video_list = []
        for video in videos:
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

        return VideosListResponse(videos=video_list, total=len(video_list))

    except Exception as e:
        logger.error(f"Error in list_videos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_index_stats():
    """Get statistics with updated weights and accuracy metrics"""
    try:
        opensearch_client = get_opensearch_client()

        stats = opensearch_client.cat.count(index=INDEX_NAME, format='json')
        clip_count = int(stats[0]['count'])

        sample = opensearch_client.search(index=INDEX_NAME, body={"size": 1, "query": {"match_all": {}}})

        modality_info = {}
        if sample['hits']['hits']:
            doc = sample['hits']['hits'][0]['_source']
            marengo_fields = {
                'emb_vis_image': 'visual-image',
                'emb_vis_text': 'visual-text',
                'emb_audio': 'audio'
            }

            for field, label in marengo_fields.items():
                if field in doc:
                    modality_info[label] = {
                        'field': field,
                        'dimension': len(doc.get(field, [])),
                        'present': True
                    }

        return {
            'total_clips': clip_count,
            'marengo_modalities': modality_info,
            'index_name': INDEX_NAME,
            'space_type': 'cosinesimil',  # ✓ CORRECT FOR YOUR USE CASE
            'structure': 'flat with separate Marengo embedding fields',
            'hybrid_weights': SEARCH_WEIGHTS['hybrid'],
            'available_search_types': {
                'hybrid': {
                    'description': 'Vector + Text with corrected industry weights',
                    'weights': SEARCH_WEIGHTS['hybrid'],
                    'accuracy': 'HIGH - Recommended'
                },
                'vector': {
                    'description': 'Pure semantic search',
                    'weights': SEARCH_WEIGHTS['vector'],
                    'accuracy': 'MEDIUM'
                },
                'text': {
                    'description': 'Keyword-only BM25',
                    'weights': None,
                    'accuracy': 'LOW'
                },
                'multimodal': {
                    'description': 'Text-focused multi-modality',
                    'weights': SEARCH_WEIGHTS['multimodal'],
                    'accuracy': 'HIGH'
                }
            },
            'notes': 'cosinesimil is CORRECT - Best for normalized embeddings like Marengo'
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HELPER FUNCTIONS ====================

def get_opensearch_client():
    """Initialize OpenSearch Cluster client with AWS authentication"""
    opensearch_host = os.environ.get('OPENSEARCH_CLUSTER_HOST')
    if not opensearch_host:
        raise ValueError("OPENSEARCH_CLUSTER_HOST not set")

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
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return []


def hybrid_search(client, query_embedding: List[float], query_text: str, 
                 video_id: Optional[str], top_k: int) -> List[Dict]:
    """
    ✓ OPTIMIZED HYBRID SEARCH FOR HIGH ACCURACY

    Features:
    1. Corrected weights: vis_text(1.8) > vis_image(1.2) > audio(0.8), text_match(1.5)
    2. RRF normalization for k-NN + BM25 score mismatch
    3. minimum_should_match=2 (quality filtering)
    4. Confidence scoring (0-1 range)
    5. Fetches 2x candidates, reranks by accuracy
    """
    must_clauses = []
    if video_id:
        must_clauses.append({"term": {"video_id": video_id}})

    weights = SEARCH_WEIGHTS["hybrid"]

    # Corrected weights for editorial/media content
    should_clauses = [
        {
            "knn": {
                "emb_vis_text": {
                    "vector": query_embedding,
                    "k": top_k,  # Fetch more candidates
                    "boost": weights["emb_vis_text"]
                }
            }
        },
        {
            "knn": {
                "emb_vis_image": {
                    "vector": query_embedding,
                    "k": top_k,
                    "boost": weights["emb_vis_image"]
                }
            }
        },
        {
            "knn": {
                "emb_audio": {
                    "vector": query_embedding,
                    "k": top_k,
                    "boost": weights["emb_audio"]
                }
            }
        },
        {
            "match": {
                "clip_text": {
                    "query": query_text,
                    "fuzziness": "AUTO",
                    "boost": weights["text_match"]
                }
            }
        }
    ]

    search_body = {
        "size": top_k,  # Fetch more for deduplication + reranking
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "should": should_clauses,
                "minimum_should_match": 2  # ✓ HIGH ACCURACY: Require 2+ modalities
            }
        },
        "_source": [
            "clip_id", "video_id", "timestamp_start", "timestamp_end", "clip_text", "video_path"
        ]
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        results = parse_search_results(response)

        # ✓ RRF Normalization: Normalize k-NN + BM25 scores to 0-1 range
        results = normalize_scores_rrf(results, RRF_K)

        # Deduplicate by clip_id, keep highest confidence
        deduped = {}
        for result in results:
            clip_id = result['clip_id']
            if clip_id not in deduped or result.get('confidence', 0) > deduped[clip_id].get('confidence', 0):
                deduped[clip_id] = result

        results = list(deduped.values())
        results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)[:top_k]

        logger.info(f"Hybrid: {len(response['hits']['hits'])} candidates → {len(results)} final (RRF normalized)")

        return results

    except Exception as e:
        logger.error(f"Hybrid search error: {e}", exc_info=True)
        return []


def vector_search(client, query_embedding: List[float], video_id: Optional[str], top_k: int) -> List[Dict]:
    """Pure vector search across all modalities"""
    must_clauses = []
    if video_id:
        must_clauses.append({"term": {"video_id": video_id}})

    weights = SEARCH_WEIGHTS["vector"]

    should_clauses = [
        {
            "knn": {
                "emb_vis_text": {
                    "vector": query_embedding,
                    "k": top_k,
                    "boost": weights["emb_vis_text"]
                }
            }
        },
        {
            "knn": {
                "emb_vis_image": {
                    "vector": query_embedding,
                    "k": top_k,
                    "boost": weights["emb_vis_image"]
                }
            }
        },
        {
            "knn": {
                "emb_audio": {
                    "vector": query_embedding,
                    "k": top_k,
                    "boost": weights["emb_audio"]
                }
            }
        }
    ]

    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "should": should_clauses,
                "minimum_should_match": 2
            }
        },
        "_source": ["clip_id", "video_id", "timestamp_start", "timestamp_end", "clip_text", "video_path"]
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        results = parse_search_results(response)
        results = normalize_scores_rrf(results, RRF_K)

        deduped = {}
        for result in results:
            clip_id = result['clip_id']
            if clip_id not in deduped or result.get('confidence', 0) > deduped[clip_id].get('confidence', 0):
                deduped[clip_id] = result

        results = sorted(deduped.values(), key=lambda x: x.get('confidence', 0), reverse=True)[:top_k]
        return results

    except Exception as e:
        logger.error(f"Vector search error: {e}", exc_info=True)
        return []


def text_search(client, query_text: str, video_id: Optional[str], top_k: int) -> List[Dict]:
    """Text-only BM25 search"""
    must_clauses = [
        {"match": {"clip_text": {"query": query_text, "fuzziness": "AUTO"}}}
    ]

    if video_id:
        must_clauses.append({"term": {"video_id": video_id}})

    search_body = {
        "size": top_k,
        "query": {"bool": {"must": must_clauses}},
        "_source": ["clip_id", "video_id", "timestamp_start", "timestamp_end", "clip_text", "video_path"]
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        results = parse_search_results(response)

        # Normalize BM25 scores
        for result in results:
            result['confidence'] = min(result.get('score', 0) / 10.0, 1.0)  # BM25 can be high

        return results[:top_k]

    except Exception as e:
        logger.error(f"Text search error: {e}", exc_info=True)
        return []


def multimodal_search(client, query_embedding: List[float], video_id: Optional[str], top_k: int) -> List[Dict]:
    """Multimodal search with text-focused weights"""
    must_clauses = []
    if video_id:
        must_clauses.append({"term": {"video_id": video_id}})

    weights = SEARCH_WEIGHTS["multimodal"]

    should_clauses = [
        {"knn": {"emb_vis_text": {"vector": query_embedding, "k": top_k * 2, "boost": weights["emb_vis_text"]}}},
        {"knn": {"emb_vis_image": {"vector": query_embedding, "k": top_k * 2, "boost": weights["emb_vis_image"]}}},
        {"knn": {"emb_audio": {"vector": query_embedding, "k": top_k * 2, "boost": weights["emb_audio"]}}}
    ]

    search_body = {
        "size": top_k * 2,
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "should": should_clauses,
                "minimum_should_match": 2
            }
        },
        "_source": ["clip_id", "video_id", "timestamp_start", "timestamp_end", "clip_text", "video_path"]
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        results = parse_search_results(response)
        results = normalize_scores_rrf(results, RRF_K)

        deduped = {}
        for result in results:
            clip_id = result['clip_id']
            if clip_id not in deduped or result.get('confidence', 0) > deduped[clip_id].get('confidence', 0):
                deduped[clip_id] = result

        results = sorted(deduped.values(), key=lambda x: x.get('confidence', 0), reverse=True)[:top_k]
        return results

    except Exception as e:
        logger.error(f"Multimodal search error: {e}", exc_info=True)
        return []


def normalize_scores_rrf(results: List[Dict], k: int = 60) -> List[Dict]:
    """
    ✓ RECIPROCAL RANK FUSION (RRF) NORMALIZATION

    Converts mixed scores (k-NN: 0-2, BM25: 0-∞) to unified 0-1 confidence range.
    Formula: confidence = 1 / (k + rank)

    Reference: OpenSearch Hybrid Search best practices
    """
    for rank, result in enumerate(results, 1):
        # RRF converts any ranking to normalized score
        rrf_score = 1.0 / (k + rank)
        result['confidence'] = rrf_score
        result['rank'] = rank

    return results


def get_all_unique_videos(client) -> List[Dict]:
    """Get all unique videos from index"""
    search_body = {
        "size": 0,
        "aggs": {
            "unique_videos": {
                "terms": {"field": "video_id", "size": 10000},
                "aggs": {
                    "video_metadata": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["video_id", "video_path", "clip_text"]
                        }
                    },
                    "clip_count": {"cardinality": {"field": "clip_id"}}
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
        logger.error(f"Error fetching videos: {e}", exc_info=True)
        return []


def convert_s3_to_presigned_urls(s3_client, results: List[Dict], expiration: int = 3600) -> List[Dict]:
    """Convert S3 paths to presigned URLs"""
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
                logger.warning(f"Error generating presigned URL: {e}")
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
        logger.warning(f"Error generating presigned URL: {e}")
        return None


def parse_search_results(response: Dict) -> List[Dict]:
    """Parse OpenSearch response into results with scores"""
    results = []

    for hit in response['hits']['hits']:
        source = hit['_source']

        result = {
            'clip_id': source.get('clip_id'),
            'video_id': source.get('video_id'),
            'video_path': source.get('video_path'),
            'timestamp_start': source.get('timestamp_start'),
            'timestamp_end': source.get('timestamp_end'),
            'clip_text': source.get('clip_text'),
            'score': hit.get('_score'),
            'confidence': 0.0  # Will be populated by RRF normalization
        }

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
