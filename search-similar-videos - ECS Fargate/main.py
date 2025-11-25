import math
from fastapi import FastAPI, HTTPException
import json
import boto3
import os
import logging
import base64
import uuid
import datetime
import asyncio
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict, Optional
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Video Search Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://condenast-fe.s3-website-us-east-1.amazonaws.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CHANGE 1: Updated index name to consolidated index
INDEX_NAME = "video_clips_consolidated"
VECTOR_PIPELINE = "vector-norm-pipeline-consolidated-index-rrf"
VECTOR_PIPELINE_3_VECTOR = "vector-norm-pipeline-video-clips-3-vector-rrf"
MIN_SCORE = 0.5
INNER_MIN_SCORE_VISUAL = INNER_MIN_SCORE_AUDIO = INNER_MIN_SCORE_TRANSCRIPTION = INNER_MIN_SCORE = 0.6
INNER_TOP_K = 100
TOP_K = 50

# Intent-based search pipelines for Marengo 3
VECTOR_PIPELINE_3_VISUAL = "vector-norm-pipeline-video-clips-3-visual-intent"
VECTOR_PIPELINE_3_AUDIO = "vector-norm-pipeline-video-clips-3-audio-intent"
VECTOR_PIPELINE_3_TRANSCRIPT = "vector-norm-pipeline-video-clips-3-text-intent"
VECTOR_PIPELINE_3_BALANCED = "vector-norm-pipeline-video-clips-3-balanced-intent"

# Intent-to-weights mapping for RRF pipeline (visual, audio, transcription)
INTENT_WEIGHTS = {
    "VISUAL":   [0.8, 0.1, 0.1],   # visual-focused
    "AUDIO":    [0.1, 0.7, 0.2],   # audio-focused
    "TRANSCRIPT": [0.1, 0.2, 0.7], # text-focused
    "BALANCED": [0.34, 0.33, 0.33] # balanced across all
}

# Initialize clients at startup
opensearch_client = None
bedrock_runtime = None
s3_client = None
vector_pipeline_exists = False
hybrid_pipeline_exists = False


@app.on_event("startup")
async def startup_event():
    """Initialize clients and pipelines on application startup"""
    global opensearch_client, bedrock_runtime, s3_client, vector_pipeline_exists, hybrid_pipeline_exists

    try:
        logger.info("Initializing clients...")
        # logger.info("1")
        opensearch_client = get_opensearch_client()
        bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
        s3_client = boto3.client("s3", region_name="us-east-1")

        logger.info("Initializing search pipelines...")
        hybrid_pipeline_exists = _create_hybrid_search_pipeline(opensearch_client)
        vector_pipeline_exists = _create_vector_search_pipeline(opensearch_client)
        _create_vector_search_pipeline_3_vector(opensearch_client)

        # Create intent-based pipelines for Marengo 3
        logger.info("Creating intent-based search pipelines...")
        _create_intent_based_pipelines(opensearch_client)

        # logger.info("Configuring S3 CORS policy...")
        # _configure_s3_cors(s3_client)

        logger.info("✓ All clients and pipelines initialized successfully")
    except Exception as e:
        logger.error(f"✗ Startup initialization failed: {e}", exc_info=True)
        raise


class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    image_base64: Optional[str] = None
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
    classified_intent: Optional[str] = None  
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
    Unified search endpoint - handles both text and image searches
    - Text search: Uses query_text with specified search_type
    - Image search: Uses image_base64, generates embedding, performs image-specific search
    """
    try:
        query_text = request.query_text
        image_base64 = request.image_base64
        top_k = request.top_k
        search_type = request.search_type
        INDEX_NAME = "video_clips_consolidated"

        # Validate that at least one input is provided
        if not query_text and not image_base64:
            raise HTTPException(
                status_code=400, detail="Either query_text or image_base64 is required"
            )

        # IMAGE SEARCH PATH
        if image_base64:
            logger.info(f"📷 Image search requested (top_k: {top_k})")

            # Validate image
            is_valid, error_msg = validate_image(image_base64)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            logger.info("✓ Image validation passed")
            logger.info(
                f"Processing image base64 of length: {len(image_base64)} characters"
            )

            # Generate image embedding
            logger.info("🔄 Generating image embedding from base64 using Marengo")
            query_embedding = generate_image_embedding(bedrock_runtime, image_base64)

            if not query_embedding:
                raise HTTPException(
                    status_code=500, detail="Failed to generate image embedding"
                )

            logger.info(
                f"✓ Generated image embedding with {len(query_embedding)} dimensions"
            )

            # Perform image-specific search
            logger.info("🔍 Performing image-specific search using emb_vis_image")
            results = search_with_image(
                opensearch_client, query_embedding, top_k, INDEX_NAME
            )

            query_display = ""  # Empty query for image search
            search_type_display = "image"

        # TRANSCRIPT SEARCH PATH
        else:
            logger.info(
                f"🔍 Text search: '{query_text}' (type: {search_type}, top_k: {top_k})"
            )
            logger.info("Generating embedding from text using Marengo")
            query_embedding = generate_text_embedding(bedrock_runtime, str(query_text))

            if not query_embedding:
                raise HTTPException(
                    status_code=500, detail="Failed to generate query embedding"
                )

            logger.info(
                f"Generated text embedding with {len(query_embedding)} dimensions"
            )

            # Perform search based on type for text queries
            if search_type == "hybrid":
                results = hybrid_search(
                    opensearch_client,
                    query_embedding,
                    str(query_text),
                    top_k,
                    INDEX_NAME,
                )
            elif search_type == "vector":
                results = vector_search(
                    opensearch_client, query_embedding, top_k, INDEX_NAME
                )
            elif search_type == "visual":
                results = visual_search(
                    opensearch_client, query_embedding, top_k, INDEX_NAME
                )
            elif search_type == "audio":
                results = audio_search(
                    opensearch_client, query_embedding, top_k, INDEX_NAME
                )
            # elif search_type == 'text':
            #     results = text_search(opensearch_client, query_text, top_k)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Invalid search_type: {search_type}"
                )

            query_display = query_text
            search_type_display = search_type

        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)

        logger.info(f"✓ Search completed, found {len(results)} results")

        return SearchResponse(
            query=str(query_display),
            search_type=search_type_display,
            total=len(results),
            clips=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-3", response_model=SearchResponse)
async def search_videos_marengo3(request: SearchRequest):
    """
    Marengo 3 unified search endpoint with intent classification
    - Text only: Classifies intent first, then generates embedding via Marengo 3
    - Image only: Uses image_base64, generates embedding via Marengo 3
    - Combined: Uses both query_text and image_base64 for multimodal search

    Intent classification (for text queries):
    - VISUAL: Focus on visual embeddings
    - AUDIO: Focus on audio embeddings
    - TRANSCRIPT: Focus on transcription embeddings
    - BALANCED: Use all three with balanced weights
    """
    try:
        query_text = request.query_text
        image_base64 = request.image_base64
        top_k = request.top_k
        search_type = request.search_type

        # Validate that at least one input is provided
        if not query_text and not image_base64:
            raise HTTPException(
                status_code=400, detail="Either query_text or image_base64 is required"
            )

        # Validate image if provided
        if image_base64:
            is_valid, error_msg = validate_image(image_base64)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            logger.info("✓ Image validation passed")

        # Determine search type for logging
        if query_text and image_base64:
            logger.info(
                f"🔄 Multimodal search (Marengo 3): text='{query_text[:50]}...' + image (top_k: {top_k})"
            )
            search_input_type = "multimodal"
        elif image_base64:
            logger.info(f"📷 Image-only search (Marengo 3) (top_k: {top_k})")
            search_input_type = "image"
        else:
            logger.info(
                f"🔍 Text-only search (Marengo 3): '{query_text}' (type: {search_type}, top_k: {top_k})"
            )
            search_input_type = "text"

        # STEP 1 & 2: Run intent classification and embedding generation CONCURRENTLY
        classified_intent = None
        query_embedding = None

        if query_text and not image_base64 and search_type == "vector":
            # For text-only vector search: Run BOTH intent classification and embedding generation in parallel
            logger.info(
                "📊 Step 1 & 2: Running intent classification and embedding generation concurrently..."
            )

            # Create both tasks
            intent_task = classify_query_intent(bedrock_runtime, query_text)
            embedding_task = asyncio.to_thread(
                generate_embedding_marengo3,
                bedrock_runtime,
                text=query_text,
                image_base64=image_base64,
            )

            # Run both concurrently and wait for both to complete
            classified_intent, query_embedding = await asyncio.gather(
                intent_task, embedding_task
            )

            logger.info(f"✓ Intent classification result: {classified_intent}")
            logger.info(
                f"✓ Generated {search_input_type} embedding (Marengo 3) with {len(query_embedding) if query_embedding else 0} dimensions"
            )

            # Map intent to search type
            intent_based_search_type = get_search_type_from_intent(classified_intent)
            logger.info(
                f"📊 Mapped intent '{classified_intent}' to search_type: '{intent_based_search_type}'"
            )
        else:
            # For other cases (image, multimodal, or direct modality search): Only generate embedding
            logger.info(
                f"📊 Step 2: Generating {search_input_type} embedding using Marengo 3"
            )
            query_embedding = generate_embedding_marengo3(
                bedrock_runtime, text=query_text, image_base64=image_base64
            )
            logger.info(
                f"✓ Generated {search_input_type} embedding (Marengo 3) with {len(query_embedding) if query_embedding else 0} dimensions"
            )

        if not query_embedding:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate {search_input_type} embedding (Marengo 3)",
            )

        # STEP 3: Perform search based on type
        logger.info(f"📊 Step 3: Performing {search_type} search (Marengo 3)")

        if search_type == "hybrid":
            logger.info(
                "⚠️ Hybrid search not yet implemented for Marengo 3, using vector search instead"
            )
            results = vector_search_marengo3(
                opensearch_client, query_embedding, top_k, "video_clips_3_lucene"
            )
        elif search_type == "vector":
            # For vector search, use intent classification if available (text-only queries)
            if classified_intent:
                logger.info(
                    f"📊 Using intent-based vector search with intent: {classified_intent}"
                )
                results = vector_search_marengo3_with_intent(
                    opensearch_client,
                    query_embedding,
                    classified_intent,
                    top_k,
                    "video_clips_3_lucene",
                )
            else:
                # For image or multimodal queries, use balanced weights
                logger.info(
                    "📊 Using balanced vector search (image or multimodal query)"
                )
                results = vector_search_marengo3(
                    opensearch_client, query_embedding, top_k, "video_clips_3_lucene"
                )
        elif search_type == "visual":
            results = visual_search_marengo3(
                opensearch_client, query_embedding, top_k, "video_clips_3_lucene"
            )
        elif search_type == "audio":
            results = audio_search_marengo3(
                opensearch_client, query_embedding, top_k, "video_clips_3_lucene"
            )
        elif search_type == "transcription":
            results = transcription_search_marengo3(
                opensearch_client, query_embedding, top_k, "video_clips_3_lucene"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_type: {search_type}. Supported: vector, visual, audio, transcription",
            )

        query_display = query_text if query_text else ""
        search_type_display = search_type

        # Convert S3 paths to presigned URLs
        results = convert_s3_to_presigned_urls(s3_client, results)

        logger.info(f"✓ Search (Marengo 3) completed, found {len(results)} results")

        return SearchResponse(
            query=query_display,
            classified_intent=classified_intent,
            search_type=search_type_display,
            total=len(results),
            clips=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search-3: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=VideosListResponse)
async def list_all_videos():
    """
    Get all unique videos from the OpenSearch index
    Returns video metadata including S3 paths and clip counts
    """
    try:
        # Get all unique videos from OpenSearch
        videos = get_all_unique_videos(opensearch_client)

        # Transform to response format
        video_list = []
        for video in videos:
            # Generate presigned URL for private S3 bucket access
            presigned_url = convert_s3_to_presigned_url(s3_client, video["video_path"])

            video_list.append(
                VideoMetadata(
                    video_id=video["video_id"],
                    video_path=presigned_url if presigned_url else video["video_path"],
                    title=video.get("clip_text") or f"Video {video['video_id'][:8]}",
                    thumbnail_url=video.get("thumbnail_url"),
                    duration=video.get("duration"),
                    upload_date=video.get("upload_date"),
                    clips_count=video.get("clips_count", 0),
                )
            )

        return VideosListResponse(videos=video_list, total=len(video_list))

    except Exception as e:
        logger.error(f"Error in list_videos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-upload-presigned-url")
async def generate_upload_url(filename: str):
    """
    Generate a presigned URL for direct S3 upload from frontend
    Frontend uses this URL to upload video directly to S3 without exposing credentials
    """
    try:
        # Check if s3_client is initialized
        if s3_client is None:
            raise HTTPException(
                status_code=503,
                detail="S3 client not initialized. Service may still be starting up.",
            )

        if not filename or len(filename.strip()) == 0:
            raise HTTPException(status_code=400, detail="filename is required")

        # Sanitize filename
        sanitized_name = "".join(
            c if c.isalnum() or c in ".-_" else "_" for c in filename
        )

        s3_key = f"{sanitized_name}"

        # Get bucket name from environment
        bucket_name = os.environ.get("AWS_S3_BUCKET")
        if not bucket_name:
            raise ValueError("AWS_S3_BUCKET environment variable not set")

        # Generate presigned URL for PUT operation (15 minutes expiry)
        # Include ContentType to match the Content-Type header sent by frontend
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket_name, "Key": s3_key, "ContentType": "video/mp4"},
            ExpiresIn=900,  # 15 minutes
        )

        logger.info(f"✓ Generated presigned upload URL for: {s3_key}")

        return {
            "presigned_url": presigned_url,
            "s3_key": s3_key,
            "s3_path": f"s3://{bucket_name}/{s3_key}",
            "expires_in": 900,
        }

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_opensearch_client():
    """Initialize OpenSearch Cluster client"""
    opensearch_host = os.environ.get("OPENSEARCH_CLUSTER_HOST")
    if not opensearch_host:
        raise ValueError("OPENSEARCH_CLUSTER_HOST environment variable not set")

    opensearch_host = (
        opensearch_host.replace("https://", "").replace("http://", "").strip()
    )

    session = boto3.Session()
    credentials = session.get_credentials()

    auth = AWSV4SignerAuth(credentials, "us-east-1", "es")

    return OpenSearch(
        hosts=[{"host": opensearch_host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def validate_image(image_base64: str) -> tuple[bool, str]:
    """
    Validate if the provided base64 string is a valid image
    Returns (is_valid, error_message)
    """
    try:
        # Check if base64 string is not empty
        if not image_base64 or len(image_base64.strip()) == 0:
            return False, "Image base64 string is empty"

        # Try to decode base64
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            return False, f"Invalid base64 encoding: {str(e)}"

        # Check minimum size (at least 100 bytes)
        if len(image_data) < 100:
            return False, "Image data is too small"

        # Check maximum size (5MB)
        max_size = 5 * 1024 * 1024
        if len(image_data) > max_size:
            return False, "Image data exceeds 5MB limit"

        # Validate image magic bytes (signatures)
        valid_signatures = {
            b"\xff\xd8\xff": "jpeg",
            b"\x89\x50\x4e\x47": "png",
            b"\x47\x49\x46": "gif",
            b"\x52\x49\x46\x46": "webp",
        }

        is_valid_image = False
        for signature in valid_signatures:
            if image_data.startswith(signature):
                is_valid_image = True
                break

        if not is_valid_image:
            return (
                False,
                "Image format not supported. Supported formats: JPEG, PNG, GIF, WebP",
            )

        logger.info(f"✓ Image validation passed. Size: {len(image_data)} bytes")
        return True, ""

    except Exception as e:
        logger.error(f"Error validating image: {e}", exc_info=True)
        return False, f"Image validation error: {str(e)}"


def generate_text_embedding(bedrock_runtime, text: str) -> List[float]:
    """Generate embedding for text query using Bedrock Marengo"""
    try:
        request_body = {"inputType": "text", "inputText": text, "textTruncate": "none"}

        response = bedrock_runtime.invoke_model(
            modelId="us.twelvelabs.marengo-embed-2-7-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        if "data" in result and len(result["data"]) > 0:
            return result["data"][0].get("embedding", [])

        return []

    except Exception as e:
        logger.error(f"Error generating text embedding: {e}", exc_info=True)
        return []


def generate_image_embedding(bedrock_runtime, image_base64: str) -> List[float]:
    """Generate embedding for image query using Bedrock Marengo with base64 image"""
    try:
        # Validate base64 string is not empty
        if not image_base64 or len(image_base64.strip()) == 0:
            logger.error("Image base64 string is empty")
            return []

        logger.info(
            f"Processing image base64 of length: {len(image_base64)} characters"
        )

        request_body = {
            "inputType": "image",
            "mediaSource": {"base64String": image_base64},
        }

        logger.info("Sending image embedding request to Marengo with base64 image")
        response = bedrock_runtime.invoke_model(
            modelId="us.twelvelabs.marengo-embed-2-7-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        logger.info(f"Marengo response: {result}")

        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0].get("embedding", [])
            logger.info(f"✓ Generated image embedding with {len(embedding)} dimensions")
            return embedding

        logger.warning(f"No embedding data in response. Response: {result}")
        return []

    except Exception as e:
        logger.error(f"Error generating image embedding: {e}", exc_info=True)
        return []


# ============ INTENT CLASSIFICATION FUNCTION ============


async def classify_query_intent(bedrock_runtime, query_text: str) -> str:
    """
    Classify user query intent using Bedrock Nova Micro model.
    Returns one of: VISUAL, AUDIO, TRANSCRIPT, or BALANCED
    """
    try:
        if not query_text or len(query_text.strip()) == 0:
            logger.info("Empty query, defaulting to BALANCED intent")
            return "BALANCED"

        prompt = f"""You are a modality classifier for a video library. Return EXACTLY ONE WORD from this list:
VISUAL, AUDIO, TRANSCRIPT, BALANCED.

Classify the following query into ONE of these categories:

1. VISUAL: The user is describing how something looks (objects, colors, scenes, actions).
2. AUDIO: The user is describing a sound (noises, music style, volume).
3. TRANSCRIPT: The user is searching for specific spoken words, quotes, names, or a transcript.
4. BALANCED: The query is abstract, emotional, or a mix.

Never explain. Never use punctuation. Output only one word.
INPUT: {query_text}"""

        logger.info(f"🔍 Classifying query intent: '{query_text[:50]}...'")

        request_body = {"messages": [{"role": "user", "content": [{"text": prompt}]}]}

        response = bedrock_runtime.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        logger.info(f"Nova Micro response: {result}")
        intent = (
            result.get("output", [{}])
            .get("message", [{}])
            .get("content", [{}])[0]
            .get("text", "BALANCED")
            .strip()
            .upper()
        )

        # Validate intent is one of the allowed values
        valid_intents = ["VISUAL", "AUDIO", "TRANSCRIPT", "BALANCED"]
        if intent not in valid_intents:
            logger.warning(
                f"Invalid intent '{intent}' returned, defaulting to BALANCED"
            )
            intent = "BALANCED"

        logger.info(f"✓ Query intent classified as: {intent}")
        return intent

    except Exception as e:
        logger.error(f"Error classifying query intent: {e}", exc_info=True)
        logger.info("Defaulting to BALANCED intent due to classification error")
        return "BALANCED"


def get_search_type_from_intent(intent: str) -> str:
    """
    Map intent classification to search type for weighted search
    """
    intent_to_search_type = {
        "VISUAL": "visual",
        "AUDIO": "audio",
        "TRANSCRIPT": "transcription",
        "BALANCED": "vector",     # Use all three with balanced weights
    }
    return intent_to_search_type.get(intent, "vector")


def generate_embedding_marengo3(
    bedrock_runtime, text: Optional[str] = None, image_base64: Optional[str] = None
) -> List[float]:
    """
    Generate unified embedding for Marengo 3 - supports text, image, or both
    When both are provided, Marengo 3 generates a combined multimodal embedding
    """
    try:
        # Validate at least one input is provided
        if not text and not image_base64:
            logger.error("Either text or image_base64 must be provided")
            return []

        request_body = {}

        # Text-only request
        if text and not image_base64:
            logger.info(f"🔄 Generating text embedding (Marengo 3): '{text[:50]}...'")
            request_body = {"inputType": "text", "text": {"inputText": text}}

        # Image-only request
        elif image_base64 and not text:
            if not image_base64 or len(image_base64.strip()) == 0:
                logger.error("Image base64 string is empty")
                return []

            logger.info(
                f"🔄 Generating image embedding (Marengo 3) (base64 length: {len(image_base64)} chars)"
            )
            request_body = {
                "inputType": "image",
                "image": {"mediaSource": {"base64String": image_base64}},
            }

        # Multimodal request with both text and image
        else:
            if not image_base64 or len(image_base64.strip()) == 0:
                logger.error("Image base64 string is empty")
                return []

            logger.info(f"🔄 Generating multimodal embedding (Marengo 3): text + image")
            request_body = {
                "inputType": "text_image",
                "text_image": {
                    "inputText": text,
                    "mediaSource": {"base64String": image_base64},
                },
            }

        logger.info(f"📤 Invoking Marengo 3 model")
        response = bedrock_runtime.invoke_model(
            modelId="us.twelvelabs.marengo-embed-3-0-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        logger.info(f"✓ Marengo 3 response received")

        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0].get("embedding", [])
            input_type = (
                "multimodal (text+image)"
                if (text and image_base64)
                else ("image" if image_base64 else "text")
            )
            logger.info(
                f"✓ Generated {input_type} embedding (Marengo 3) with {len(embedding)} dimensions"
            )
            return embedding

        logger.warning(f"No embedding data in Marengo 3 response. Response: {result}")
        return []

    except Exception as e:
        logger.error(f"Error generating embedding (Marengo 3): {e}", exc_info=True)
        return []


def search_with_image(
    client, query_embedding: List[float], top_k: int = 10, index_name=None
) -> List[Dict]:
    """Image-specific search using emb_vis_image field"""
    if index_name is None:
        index_name = INDEX_NAME

    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "emb_vis_image": {
                    "vector": query_embedding,
                    "min_score": INNER_MIN_SCORE_VISUAL,
                }
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    try:
        response = client.search(index=index_name, body=search_body)
        logger.info(
            f"✓ Image search completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results(response)

    except Exception as e:
        logger.error(f"Image search error: {e}", exc_info=True)
        return []


def convert_s3_to_presigned_urls(
    s3_client, results: List[Dict], expiration: int = 3600
) -> List[Dict]:
    """Convert S3 paths to presigned URLs in video_path and thumbnail_path fields"""
    for result in results:
        # Convert video_path to presigned URL
        video_path = result.get("video_path", "")
        if video_path.startswith("s3://"):
            try:
                s3_parts = video_path.replace("s3://", "").split("/", 1)
                bucket = s3_parts[0]
                key = s3_parts[1] if len(s3_parts) > 1 else ""

                presigned_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expiration,
                )

                result["video_path"] = presigned_url

            except Exception as e:
                logger.warning(f"Error generating presigned URL for {video_path}: {e}")
                pass

        # Convert thumbnail_path to presigned URL
        thumbnail_path = result.get("thumbnail_path", "")
        if thumbnail_path and thumbnail_path.startswith("s3://"):
            try:
                s3_parts = thumbnail_path.replace("s3://", "").split("/", 1)
                bucket = s3_parts[0]
                key = s3_parts[1] if len(s3_parts) > 1 else ""

                presigned_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expiration,
                )

                result["thumbnail_path"] = presigned_url
                # logger.info(f"✓ Generated presigned URL for thumbnail: {key}")

            except Exception as e:
                logger.warning(
                    f"Error generating presigned URL for thumbnail {thumbnail_path}: {e}"
                )
                pass

    return results


def convert_s3_to_presigned_url(
    s3_client, video_path: str, expiration: int = 3600
) -> Optional[str]:
    """Convert single S3 path to presigned URL"""
    if not video_path.startswith("s3://"):
        return None

    try:
        s3_parts = video_path.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        key = s3_parts[1] if len(s3_parts) > 1 else ""

        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration
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
                "terms": {"field": "video_id", "size": 10000},
                "aggs": {
                    "video_metadata": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["video_id", "video_path", "clip_text"],
                        }
                    },
                    "clip_count": {"cardinality": {"field": "clip_id"}},
                },
            }
        },
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)

        videos = []
        for bucket in response["aggregations"]["unique_videos"]["buckets"]:
            video_data = bucket["video_metadata"]["hits"]["hits"][0]["_source"]
            video_data["clips_count"] = bucket["clip_count"]["value"]
            videos.append(video_data)

        return videos

    except Exception as e:
        logger.error(f"Error fetching unique videos: {e}", exc_info=True)
        return []


# CHANGE 3: Updated hybrid_search to query emb_vis_text and emb_audio
def hybrid_search(
    client,
    query_embedding: List[float],
    query_text: str,
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_consolidated",
) -> List[Dict]:
    """Hybrid search combining vector similarity on visual-text & audio + text matching"""
    search_body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    # Visual-text embedding (k-NN) - weight 0.5
                    {"knn": {"emb_vis_text": {"vector": query_embedding, "k": top_k}}},
                    # Audio embedding (k-NN) - weight 0.3
                    {"knn": {"emb_audio": {"vector": query_embedding, "k": top_k}}},
                    # Text matching (BM25) - weight 0.2
                    {
                        "match": {
                            "video_name": {"query": query_text, "fuzziness": "AUTO"}
                        }
                    },
                ]
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    if hybrid_pipeline_exists:
        search_params = {
            "index": INDEX_NAME,
            "body": search_body,
            "search_pipeline": "hybrid-norm-pipeline",
        }
    else:
        search_params = {"index": INDEX_NAME, "body": search_body}

    try:
        response = client.search(**search_params)
        return parse_search_results(response)

    except Exception as e:
        logger.error(f"Hybrid search error: {e}", exc_info=True)
        return vector_search(client, query_embedding, top_k)


# CHANGE 4: Updated vector_search to query emb_vis_text and emb_audio
def vector_search(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_consolidated",
) -> List[Dict]:
    """Vector-only k-NN search on visual-text and audio embeddings with normalization"""
    search_body = {
        "size": TOP_K,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "emb_vis_text": {
                                "vector": query_embedding,
                                "min_score": INNER_MIN_SCORE_VISUAL,
                            }
                        }
                    },
                    {
                        "knn": {
                            "emb_audio": {
                                "vector": query_embedding,
                                "min_score": INNER_MIN_SCORE_AUDIO,
                            }
                        }
                    },
                ]
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }
    # bool query did not work with search pipeline and also it allows us to have results matching the req. no.s of sub-queries
    # (its more of an atomic approach) -- TO TRY IT AGAIN TOMORROW
    # search_body = {
    # "size": TOP_K,
    # "query": {
    #     "bool": {
    #         "should": [
    #             {
    #                 "knn": {
    #                     "emb_vis_text": {
    #                         "vector": query_embedding,
    #                         "k": 100
    #                     }
    #                 }
    #             },
    #             {
    #                 "knn": {
    #                     "emb_audio": {
    #                         "vector": query_embedding,
    #                         "k": 100
    #                     }
    #                 }
    #             }
    #         ],
    #         "minimum_should_match": 1
    #     }
    # },
    # "_source": [
    #     "video_id", "video_path", "clip_id", "timestamp_start",
    #     "timestamp_end", "clip_text", "thumbnail_path",
    #     "video_name", "clip_duration", "video_duration_sec"
    #     ]
    # }
    ################################################################

    if vector_pipeline_exists:
        search_params = {
            "index": INDEX_NAME,
            "body": search_body,
            "search_pipeline": VECTOR_PIPELINE,
        }
    else:
        search_params = {"index": INDEX_NAME, "body": search_body}

    response = client.search(**search_params)
    return parse_search_results_vector(response)


# def text_search(client, query_text: str, top_k: int = 10) -> List[Dict]:
#     """Text-only BM25 search"""
#     search_body = {
#         "size": top_k,
#         "query": {
#             "match": {
#                 "video_name": {
#                     "query": query_text,
#                     "fuzziness": "AUTO"
#                 }
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start",
#                    "timestamp_end", "clip_text",  "thumbnail_path", "video_name", "clip_duration", "video_duration_sec"]
#     }

#     response = client.search(index=INDEX_NAME, body=search_body)
#     return parse_search_results(response)


def visual_search(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_consolidated",
) -> List[Dict]:
    """Visual-only k-NN search on visual-text embeddings"""
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "emb_vis_text": {
                    "vector": query_embedding,
                    "min_score": INNER_MIN_SCORE_VISUAL,
                }
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    response = client.search(index=INDEX_NAME, body=search_body)
    return parse_search_results(response)


def audio_search(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_consolidated",
) -> List[Dict]:
    """Audio-only k-NN search on audio embeddings"""
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "emb_audio": {
                    "vector": query_embedding,
                    "min_score": INNER_MIN_SCORE_AUDIO,
                }
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    response = client.search(index=INDEX_NAME, body=search_body)
    return parse_search_results(response)


# ============ NEW SEARCH FUNCTIONS FOR MARENGO 3 (emb_visual, emb_audio, emb_transcription) ============


def vector_search_marengo3_with_intent(
    client,
    query_embedding: List[float],
    intent: str,
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_3_lucene",
) -> List[Dict]:
    """
    Vector search with intent-based weights (Marengo 3)
    Uses intent-specific search pipeline with RRF weights to focus on appropriate modality
    """
    # Map intent to pipeline
    intent_pipeline_map = {
        "VISUAL": VECTOR_PIPELINE_3_VISUAL,
        "AUDIO": VECTOR_PIPELINE_3_AUDIO,
        "TRANSCRIPT": VECTOR_PIPELINE_3_TRANSCRIPT,
        "BALANCED": VECTOR_PIPELINE_3_BALANCED,
    }

    pipeline_id = intent_pipeline_map.get(intent, VECTOR_PIPELINE_3_BALANCED)
    weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["BALANCED"])

    logger.info(
        f"📊 Using intent-based pipeline for '{intent}': weights={weights}, pipeline={pipeline_id}"
    )

    search_body = {
        "size": TOP_K,
        "query": {
            "hybrid": {
                "queries": [
                    # Visual embedding (k-NN)
                    {
                        "knn": {
                            "emb_visual": {"vector": query_embedding, "k": INNER_TOP_K}
                        }
                    },
                    # Audio embedding (k-NN)
                    {
                        "knn": {
                            "emb_audio": {"vector": query_embedding, "k": INNER_TOP_K}
                        }
                    },
                    # Transcription embedding (k-NN)
                    {
                        "knn": {
                            "emb_transcription": {
                                "vector": query_embedding,
                                "k": INNER_TOP_K,
                            }
                        }
                    },
                ]
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    search_params = {
        "index": INDEX_NAME,
        "body": search_body,
        "search_pipeline": pipeline_id,  # Use intent-specific pipeline
    }

    try:
        response = client.search(**search_params)
        logger.info(
            f"✓ Vector search with intent '{intent}' (Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results_vector(response)
    except Exception as e:
        logger.error(f"Vector search with intent (Marengo 3) error: {e}", exc_info=True)
        return []


def vector_search_marengo3(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_3_lucene",
) -> List[Dict]:
    """Vector search combining visual, audio, and transcription embeddings (Marengo 3)"""
    search_body = {
        "size": TOP_K,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "emb_visual": {"vector": query_embedding, "k": INNER_TOP_K}
                        }
                    },
                    {
                        "knn": {
                            "emb_audio": {"vector": query_embedding, "k": INNER_TOP_K}
                        }
                    },
                    {
                        "knn": {
                            "emb_transcription": {
                                "vector": query_embedding,
                                "k": INNER_TOP_K,
                            }
                        }
                    },
                ]
            }
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    if vector_pipeline_exists:
        search_params = {
            "index": INDEX_NAME,
            "body": search_body,
            "search_pipeline": VECTOR_PIPELINE_3_VECTOR,
        }
    else:
        search_params = {"index": INDEX_NAME, "body": search_body}

    try:
        response = client.search(**search_params)
        logger.info(
            f"✓ Vector search (Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results_vector(response)
    except Exception as e:
        logger.error(f"Vector search (Marengo 3) error: {e}", exc_info=True)
        return []


def visual_search_marengo3(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_3_lucene",
) -> List[Dict]:
    """Visual-only k-NN search on visual embeddings (Marengo 3)"""
    search_body = {
        "size": TOP_K,
        "query": {"knn": {"emb_visual": {"vector": query_embedding, "k": INNER_TOP_K}}},
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        logger.info(
            f"✓ Visual search (Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results(response)
    except Exception as e:
        logger.error(f"Visual search (Marengo 3) error: {e}", exc_info=True)
        return []


def audio_search_marengo3(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_3_lucene",
) -> List[Dict]:
    """Audio-only k-NN search on audio embeddings (Marengo 3)"""
    search_body = {
        "size": TOP_K,
        "query": {"knn": {"emb_audio": {"vector": query_embedding, "k": INNER_TOP_K}}},
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        logger.info(
            f"✓ Audio search (Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results(response)
    except Exception as e:
        logger.error(f"Audio search (Marengo 3) error: {e}", exc_info=True)
        return []


def transcription_search_marengo3(
    client,
    query_embedding: List[float],
    top_k: int = 10,
    INDEX_NAME: str = "video_clips_3_lucene",
) -> List[Dict]:
    """Transcription-only k-NN search on transcription embeddings (Marengo 3)"""
    search_body = {
        "size": TOP_K,
        "query": {
            "knn": {"emb_transcription": {"vector": query_embedding, "k": INNER_TOP_K}}
        },
        "_source": [
            "video_id",
            "video_path",
            "clip_id",
            "timestamp_start",
            "timestamp_end",
            "clip_text",
            "thumbnail_path",
            "video_name",
            "clip_duration",
            "video_duration_sec",
        ],
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        logger.info(
            f"✓ Transcription search (Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results"
        )
        return parse_search_results(response)
    except Exception as e:
        logger.error(f"Transcription search (Marengo 3) error: {e}", exc_info=True)
        return []


# # COMMENTED OUT: Using intent classification instead of manual combination selection
# # This function is kept for reference but not used in the search pipeline
# def vector_search_visual_audio_marengo3(client, query_embedding: List[float], top_k: int = 10, INDEX_NAME: str = 'video_clips_3_lucene') -> List[Dict]:
#     """Vector search combining visual and audio embeddings (Marengo 3)"""
#     search_body = {
#         "size": TOP_K,
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     # Visual embedding (k-NN) - weight 0.6
#                     {
#                         "knn": {
#                             "emb_visual": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     },
#                     # Audio embedding (k-NN) - weight 0.4
#                     {
#                         "knn": {
#                             "emb_audio": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start",
#                    "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration", "video_duration_sec"]
#     }
#
#     if vector_pipeline_exists:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body,
#                 "search_pipeline": VECTOR_PIPELINE
#             }
#     else:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body
#             }
#
#     try:
#         response = client.search(**search_params)
#         logger.info(f"✓ Vector search (visual+audio, Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results")
#         return parse_search_results(response)
#     except Exception as e:
#         logger.error(f"Vector search (visual+audio, Marengo 3) error: {e}", exc_info=True)
#         return []


# # COMMENTED OUT: Using intent classification instead of manual combination selection
# # This function is kept for reference but not used in the search pipeline
# def vector_search_visual_transcription_marengo3(client, query_embedding: List[float], top_k: int = 10, INDEX_NAME: str = 'video_clips_3_lucene') -> List[Dict]:
#     """Vector search combining visual and transcription embeddings (Marengo 3)"""
#     search_body = {
#         "size": TOP_K,
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     # Visual embedding (k-NN) - weight 0.6
#                     {
#                         "knn": {
#                             "emb_visual": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     },
#                     # Transcription embedding (k-NN) - weight 0.4
#                     {
#                         "knn": {
#                             "emb_transcription": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start",
#                    "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration", "video_duration_sec"]
#     }
#
#     if vector_pipeline_exists:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body,
#                 "search_pipeline": VECTOR_PIPELINE
#             }
#     else:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body
#             }
#
#     try:
#         response = client.search(**search_params)
#         logger.info(f"✓ Vector search (visual+transcription, Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results")
#         return parse_search_results(response)
#     except Exception as e:
#         logger.error(f"Vector search (visual+transcription, Marengo 3) error: {e}", exc_info=True)
#         return []


# # COMMENTED OUT: Using intent classification instead of manual combination selection
# # This function is kept for reference but not used in the search pipeline
# def vector_search_audio_transcription_marengo3(client, query_embedding: List[float], top_k: int = 10, INDEX_NAME: str = 'video_clips_3_lucene') -> List[Dict]:
#     """Vector search combining audio and transcription embeddings (Marengo 3)"""
#     search_body = {
#         "size": TOP_K,
#         "query": {
#             "hybrid": {
#                 "queries": [
#                     # Audio embedding (k-NN) - weight 0.5
#                     {
#                         "knn": {
#                             "emb_audio": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     },
#                     # Transcription embedding (k-NN) - weight 0.5
#                     {
#                         "knn": {
#                             "emb_transcription": {
#                                 "vector": query_embedding,
#                                 "k": INNER_TOP_K
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "_source": ["video_id", "video_path", "clip_id", "timestamp_start",
#                    "timestamp_end", "clip_text", "thumbnail_path", "video_name", "clip_duration", "video_duration_sec"]
#     }
#
#     if vector_pipeline_exists:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body,
#                 "search_pipeline": VECTOR_PIPELINE
#             }
#     else:
#         search_params = {
#                 "index": INDEX_NAME,
#                 "body": search_body
#             }
#
#     try:
#         response = client.search(**search_params)
#         logger.info(f"✓ Vector search (audio+transcription, Marengo 3) completed, found {len(response.get('hits', {}).get('hits', []))} results")
#         return parse_search_results(response)
#     except Exception as e:
#         logger.error(f"Vector search (audio+transcription, Marengo 3) error: {e}", exc_info=True)
#         return []


def _create_hybrid_search_pipeline(client):
    """Create search pipeline with score normalization for hybrid search"""

    pipeline_body = {
        "description": "Post-processing pipeline for hybrid search with normalization",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.5, 0.3, 0.2]},
                    },
                }
            }
        ],
    }

    try:
        client.search_pipeline.put(
            id="hybrid-norm-pipeline-consolidated-index", body=pipeline_body
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
                    "combination": {"technique": "rrf", "rank_constant": 60}
                }
            }
        ],
    }

    # pipeline_body = {
    #     "description": "Post-processing pipeline for vector search with min-max normalization (0-1 range)",
    #     "phase_results_processors": [
    #         {
    #             "normalization-processor": {
    #                 "normalization": {
    #                     "technique": "l2"
    #                 },
    #                 "combination": {
    #                     "technique": "arithmetic_mean"
    #                 }
    #             }
    #         }
    #     ]
    # }

    try:
        client.search_pipeline.put(id=VECTOR_PIPELINE, body=pipeline_body)
        logger.info("✓ Created vector search pipeline with normalization")

    except Exception as e:
        logger.warning(f"✗ Vector pipeline creation error: {e}")
        return False

    return True


def _create_intent_based_pipelines(client):
    """Create intent-based search pipelines with different RRF weights for Marengo 3"""

    intent_pipelines = {
        "VISUAL": VECTOR_PIPELINE_3_VISUAL,
        "AUDIO": VECTOR_PIPELINE_3_AUDIO,
        "TRANSCRIPT": VECTOR_PIPELINE_3_TRANSCRIPT,
        "BALANCED": VECTOR_PIPELINE_3_BALANCED,
    }

    for intent, pipeline_id in intent_pipelines.items():
        weights = INTENT_WEIGHTS[intent]

        pipeline_body = {
            "description": f"Post processor for hybrid RRF search with {intent} intent weights",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": 60,
                            "parameters": {"weights": weights},
                        }
                    }
                }
            ],
        }

        try:
            client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
            logger.info(
                f"✓ Created {intent} intent search pipeline: {pipeline_id} with weights {weights}"
            )
        except Exception as e:
            logger.warning(f"✗ {intent} intent pipeline creation error: {e}")


def _create_vector_search_pipeline_3_vector(client):
    """Create search pipeline with score normalization for vector search"""

    pipeline_body = {
        "description": "Post processor for hybrid RRF search",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 60,
                        "parameters": {"weights": [0.5, 0.4, 0.1]},
                    }
                }
            }
        ],
    }

    try:
        client.search_pipeline.put(id=VECTOR_PIPELINE_3_VECTOR, body=pipeline_body)
        logger.info("✓ Created marengo-3-vector search pipeline with normalization")

    except Exception as e:
        logger.warning(f"✗ Vector pipeline creation error: {e}")
        return False

    return True


def parse_search_results(response: Dict) -> List[Dict]:
    """Parse OpenSearch response into results list"""
    results = []

    for hit in response["hits"]["hits"]:
        result = hit["_source"]
        result["score"] = hit["_score"]
        result["_id"] = hit["_id"]
        # logger.info(result)
        results.append(result)

    return results


# def parse_search_results_vector(response: Dict) -> List[Dict]:
#     """Parse OpenSearch results and apply L2 score normalization (optimized)."""
#     results = []
#     sum_sq = 0.0

#     # First loop: collect results & accumulate squared scores
#     for hit in response['hits']['hits']:
#         result = hit['_source']

#         score = hit['_score']
#         sum_sq += score * score

#         result['score'] = score
#         result['_id'] = hit['_id']
#         results.append(result)

#     print(results)

#     # Compute L2 norm
#     norm = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
#     logger.info(f"L2 norm: {norm}")
#     # Second loop: apply normalized score
#     for r in results:
#         r['score'] = r['score'] / norm

#     print(results)

#     return results

# def parse_search_results_vector(response: Dict) -> List[Dict]:
#     """Parse OpenSearch results and apply min-max score normalization."""
#     results = []
#     scores = []

#     # First loop: collect results & raw scores
#     for hit in response['hits']['hits']:
#         result = hit['_source']
#         score = hit['_score']

#         result['score'] = score
#         result['_id'] = hit['_id']

#         results.append(result)
#         scores.append(score)

#     print(results)

#     # Compute min and max
#     if scores:
#         mn = min(scores)
#         mx = max(scores)
#     else:
#         mn, mx = 0.0, 1.0

#     logger.info(f"Min score: {mn}, Max score: {mx}")

#     # Avoid division-by-zero: if all scores equal
#     denom = mx - mn if mx != mn else 1.0

#     # Second loop: apply min-max normalization
#     for r in results:
#         r['score'] = (r['score'] - mn) / denom

#     print(results)

#     return results

# def sigmoid(x: float) -> float:
#     return 1.0 / (1.0 + math.exp(-x))

# def parse_search_results_vector(response: Dict) -> List[Dict]:
#     """Parse OpenSearch results and apply sigmoid normalization to scores."""
#     results = []
#     scores = []

#     # First loop: collect results & raw scores
#     for hit in response['hits']['hits']:
#         result = hit['_source']
#         score = hit['_score']

#         result['_id'] = hit['_id']
#         result['score_raw'] = score

#         results.append(result)
#         scores.append(score)

#     # Compute mean for centering (b)
#     if scores:
#         mean_score = sum(scores) / len(scores)
#     else:
#         mean_score = 0.0

#     final_results = []
#     # Steepness factor
#     a = 5.0   # Feel free to tune this (3–10)
#     scale = 100.0/2
#     # print(results)
#     # Apply sigmoid normalization
#     for r in results:
#         centered = (r['score_raw'] - mean_score) * scale
#         r['score'] = sigmoid(a * centered)
#         if r['score'] < MIN_SCORE:
#             break
#         final_results.append(r)

#     # print(final_results)
#     return final_results


def normalize_rrf(rrf_raw, M=1.23, k=60):
    rrf_max = M * (1.0 / (k + 1.0))  # = ~0.03278688 when M=2
    return min(1.0, rrf_raw / rrf_max)


def parse_search_results_vector(response):
    results = []

    for hit in response["hits"]["hits"]:
        raw = hit["_score"]

        result = hit["_source"]
        result["_id"] = hit["_id"]
        result["score_raw"] = raw
        result["score"] = round(normalize_rrf(raw), 3)

        results.append(result)

    # print(results)

    return results


# def _configure_s3_cors(s3_client):
#     """Configure restrictive CORS policy on S3 bucket for video uploads"""
#     bucket_name = os.environ.get('AWS_S3_BUCKET')
#     if not bucket_name:
#         logger.warning("AWS_S3_BUCKET not set, skipping CORS configuration")
#         return

#     cors_config = {
#         'CORSRules': [
#             {
#                 'AllowedOrigins': [
#                     'http://localhost:3000',
#                     'http://condenast-fe.s3-website-us-east-1.amazonaws.com',
#                     'https://condenast-fe.s3-website-us-east-1.amazonaws.com'
#                 ],
#                 'AllowedMethods': ['PUT', 'POST', 'GET', 'HEAD'],
#                 'AllowedHeaders': ['*'],
#                 'MaxAgeSeconds': 3000,
#                 'ExposeHeaders': ['ETag', 'x-amz-version-id']
#             }
#         ]
#     }

#     try:
#         s3_client.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_config)
#         logger.info(f"✓ S3 CORS policy configured for bucket: {bucket_name}")
#     except Exception as e:
#         logger.warning(f"Could not configure S3 CORS (may require manual setup): {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
