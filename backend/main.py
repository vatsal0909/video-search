from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
# from agent import VideoSearchAgent
from clients import (
    BedrockClient,
    # TwelveLabsClient,
    # OpenSearchClient,
    BedrockTwelveLabsClient,
    BedrockOpensearchClient
)
from video_routes import router as video_router
from s3_utils import add_presigned_urls_to_results

import time

app = FastAPI(title="Video Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
bedrock_client = BedrockClient()
if os.getenv('Bedrock_TL') == "True":
    twelvelabs_client = BedrockTwelveLabsClient()
    opensearch_client = BedrockOpensearchClient()
# else:
#     twelvelabs_client = TwelveLabsClient()
#     opensearch_client = OpenSearchClient()
# video_agent = VideoSearchAgent()

# Include video routes (will use the opensearch_client directly)
app.include_router(video_router)

# Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class QuestionRequest(BaseModel):
    question: str

class ProcessVideoRequest(BaseModel):
    video_url: str

class SearchResponse(BaseModel):
    results: List[dict]
    total: int

class AnswerResponse(BaseModel):
    answer: str
    clips: List[dict]

class VideoClip(BaseModel):
    video_id: str
    video_path: str
    presigned_url: Optional[str] = None
    timestamp_start: float
    timestamp_end: float
    clip_text: str
    score: float

class ClipsResponse(BaseModel):
    clips: List[VideoClip]
    total: int
    query: str

# In-memory job tracking (use Redis/DB in production)
processing_jobs = {}

def process_video_embeddings(video_id: str, video_path: str):
    """Background task to process video embeddings"""
    start_time = time.time()
    try:
        processing_jobs[video_id] = {"status": "processing", "progress": 0}
        # # Split the URL to encode only the path part
        # from urllib.parse import urlparse, urlunparse, quote
        
        # parsed = urlparse(video_path)
        # # Encode the path, but keep the slashes
        # encoded_path = quote(parsed.path, safe='/')
        
        # # Reconstruct the URL
        # encoded_url = urlunparse((
        #     parsed.scheme,
        #     parsed.netloc,
        #     encoded_path,
        #     parsed.params,
        #     parsed.query,
        #     parsed.fragment
        # ))
        
        # print(f"Using encoded URL: {encoded_url}")
        print(f"Using URL: {video_path}")
        
        # Generate embeddings via TwelveLabs
        embeddings = twelvelabs_client.generate_video_embeddings(video_path)
        
        if not embeddings:
            processing_jobs[video_id] = {"status": "failed", "error": "No embeddings generated"}
            return
        
        # Index each clip embedding
        indexed = 0
        for emb in embeddings:
            opensearch_client.index_clip(
                video_id=video_id,
                video_path=video_path,
                timestamp_start=emb['start_offset_sec'],
                timestamp_end=emb['end_offset_sec'],
                embedding_scope=emb['embedding_scope'],
                embedding=emb['embedding'],
                clip_text=f"Clip at {emb['start_offset_sec']:.1f}s"
            )
            indexed += 1
            processing_jobs[video_id]["progress"] = int((indexed / len(embeddings)) * 100)
        
        print(f"FINISHED: Indexed {indexed} video embeddings")
        end_time = time.time()
        print(f"Time taken to index video embeddings: {end_time - start_time} seconds")

        processing_jobs[video_id] = {
            "status": "completed",
            "clips_indexed": indexed,
            "progress": 100
        }
        
    except Exception as e:
        processing_jobs[video_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {"message": "Video Search API with TwelveLabs + Bedrock"}

@app.post("/process-video")
async def process_video(background_tasks: BackgroundTasks, request: ProcessVideoRequest):
    """Upload and process video"""
    try:
        # Saved video on S3
        video_id = str(uuid.uuid4())
        video_path = request.video_url
        
        # Start background processing
        background_tasks.add_task(process_video_embeddings, video_id, video_path)
        
        processing_jobs[video_id] = {"status": "queued", "progress": 0}
        
        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Video uploaded, processing started in background"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    """Check video processing status"""
    if video_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return processing_jobs[video_id]

@app.post("/hybrid-search", response_model=ClipsResponse)
async def hybrid_search(request: SearchRequest):
    """Search for video clips using TwelveLabs embeddings"""
    try:
        # Generate query embedding via TwelveLabs
        query_embedding = twelvelabs_client.generate_text_embedding(request.query)

        print("Query embedding received")
        
        if not query_embedding:
            return ClipsResponse(clips=[], total=0, query=request.query)
        
        # Search in local OpenSearch using hybrid search
        results = opensearch_client.hybrid_search(
            query_embedding,
            request.query,
            top_k=request.top_k
        )
        
        # Add presigned URLs for private S3 buckets
        results = add_presigned_urls_to_results(results, expiration=3600)
        
        # Convert results to VideoClip models
        clips = [
            VideoClip(
                video_id=result['video_id'],
                video_path=result['video_path'],
                presigned_url=result.get('presigned_url'),
                timestamp_start=result['timestamp_start'],
                timestamp_end=result['timestamp_end'],
                clip_text=result.get('clip_text', ''),
                score=result.get('score', 0.0)
            )
            for result in results
        ]

        return ClipsResponse(
            clips=clips,
            total=len(clips),
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ask", response_model=AnswerResponse)
# async def ask_question(request: QuestionRequest):
#     """Ask a question using Strands agent with TwelveLabs + Bedrock"""
#     try:
#         result = video_agent.answer_question(request.question)
        
#         return AnswerResponse(
#             answer=result['answer'],
#             clips=result['clips']
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-clips", response_model=ClipsResponse)
async def search_clips(request: SearchRequest):
    """Search for video clips with timestamps matching the query"""
    try:
        # Generate query embedding via TwelveLabs
        query_embedding = twelvelabs_client.generate_text_embedding(request.query)
        
        if not query_embedding:
            return ClipsResponse(clips=[], total=0, query=request.query)
        
        # Search in local OpenSearch
        results = opensearch_client.search_similar_clips(
            query_embedding,
            top_k=request.top_k
        )
        
        # Add presigned URLs for private S3 buckets
        results = add_presigned_urls_to_results(results, expiration=3600)
        
        # Convert results to VideoClip models
        clips = [
            VideoClip(
                video_id=result['video_id'],
                video_path=result['video_path'],
                presigned_url=result.get('presigned_url'),
                timestamp_start=result['timestamp_start'],
                timestamp_end=result['timestamp_end'],
                clip_text=result.get('clip_text', ''),
                score=result.get('score', 0.0)
            )
            for result in results
        ]
        
        return ClipsResponse(
            clips=clips,
            total=len(clips),
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    opensearch_ok = opensearch_client.client.ping()
    
    # Test TwelveLabs connection
    try:
        twelvelabs_client.generate_text_embedding("test")
        twelvelabs_ok = True
    except:
        twelvelabs_ok = False
    
    return {
        "status": "healthy" if (opensearch_ok and twelvelabs_ok) else "degraded",
        "opensearch": opensearch_ok,
        "twelvelabs": twelvelabs_ok,
        "bedrock": "configured"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
