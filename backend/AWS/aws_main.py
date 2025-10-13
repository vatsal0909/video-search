from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from aws_clients import AWSBedrockClient, AWSMarengoClient, AWSOpenSearchClient

app = FastAPI(title="AWS Video Search API with Marengo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS clients
bedrock_client = AWSBedrockClient()
marengo_client = AWSMarengoClient()
opensearch_client = AWSOpenSearchClient()

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
    timestamp_start: float
    timestamp_end: float
    clip_text: str
    score: float

class ClipsResponse(BaseModel):
    clips: List[VideoClip]
    total: int
    query: str

# In-memory job tracking (use DynamoDB in production)
processing_jobs = {}

def process_video_embeddings(video_id: str, video_path: str):
    """
    Background task to process video embeddings using AWS Bedrock Marengo.
    """
    try:
        processing_jobs[video_id] = {"status": "processing", "progress": 0}
        
        print(f"Processing video: {video_path}")
        
        # Generate embeddings via AWS Bedrock Marengo
        embeddings = marengo_client.generate_video_embeddings(video_path)
        
        if not embeddings:
            processing_jobs[video_id] = {
                "status": "failed", 
                "error": "No embeddings generated from Marengo model."
            }
            return
        
        # Index each clip embedding
        indexed = 0
        for emb in embeddings:
            opensearch_client.index_clip(
                video_id=video_id,
                video_path=video_path,
                timestamp_start=emb['start_offset_sec'],
                timestamp_end=emb['end_offset_sec'],
                embedding=emb['embedding'],
                clip_text=f"Clip at {emb['start_offset_sec']:.1f}s"
            )
            indexed += 1
            processing_jobs[video_id]["progress"] = int((indexed / len(embeddings)) * 100)
        
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
    return {"message": "AWS Video Search API with Bedrock Marengo + OpenSearch"}

@app.post("/process-video")
async def process_video(background_tasks: BackgroundTasks, request: ProcessVideoRequest):
    """
    Upload and process video using AWS Bedrock Marengo.
    """
    try:
        video_id = str(uuid.uuid4())
        video_path = request.video_url
        
        # Start background processing
        background_tasks.add_task(process_video_embeddings, video_id, video_path)
        
        processing_jobs[video_id] = {"status": "queued", "progress": 0}
        
        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Video uploaded, processing started with Bedrock Marengo"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    """Check video processing status"""
    if video_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return processing_jobs[video_id]

@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """Search for video clips using AWS Bedrock Marengo embeddings"""
    try:
        # Generate query embedding via AWS Bedrock Marengo
        query_embedding = marengo_client.generate_text_embedding(request.query)

        print(f"Generated embedding with dimension: {len(query_embedding)}")
        
        if not query_embedding:
            return SearchResponse(results=[], total=0)
        
        # Search in AWS OpenSearch
        results = opensearch_client.search_similar_clips(
            query_embedding,
            top_k=request.top_k
        )
        
        return SearchResponse(
            results=results,
            total=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question using AWS Bedrock Marengo + LLM"""
    try:
        # Generate query embedding via Marengo
        query_embedding = marengo_client.generate_text_embedding(request.question)
        
        if not query_embedding:
            return AnswerResponse(
                answer="Could not generate query embedding.",
                clips=[]
            )
        
        # Search for relevant clips
        clips = opensearch_client.search_similar_clips(query_embedding, top_k=5)
        
        if not clips:
            return AnswerResponse(
                answer="No relevant video clips found.",
                clips=[]
            )
        
        # Format context for LLM
        context = _format_clips_for_llm(clips)
        
        # Generate answer via Bedrock LLM
        answer = bedrock_client.generate_answer(context, request.question)
        
        return AnswerResponse(
            answer=answer,
            clips=clips
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-clips", response_model=ClipsResponse)
async def search_clips(request: SearchRequest):
    """Search for video clips with timestamps matching the query"""
    try:
        # Generate query embedding via AWS Bedrock Marengo
        query_embedding = marengo_client.generate_text_embedding(request.query)
        
        if not query_embedding:
            return ClipsResponse(clips=[], total=0, query=request.query)
        
        # Search in AWS OpenSearch
        results = opensearch_client.search_similar_clips(
            query_embedding,
            top_k=request.top_k
        )
        
        # Convert results to VideoClip models
        clips = [
            VideoClip(
                video_id=result['video_id'],
                video_path=result['video_path'],
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
    
    # Test AWS Bedrock Marengo connection
    try:
        marengo_client.generate_text_embedding("test")
        marengo_ok = True
    except:
        marengo_ok = False
    
    return {
        "status": "healthy" if (opensearch_ok and marengo_ok) else "degraded",
        "opensearch": opensearch_ok,
        "marengo_embeddings": marengo_ok,
        "bedrock_llm": "configured"
    }

def _format_clips_for_llm(clips: List[Dict]) -> str:
    """Format clips for LLM context"""
    formatted = []
    for i, clip in enumerate(clips, 1):
        formatted.append(
            f"Clip {i}:\n"
            f"  Video: {clip['video_id']}\n"
            f"  Time: {clip['timestamp_start']:.1f}s - {clip['timestamp_end']:.1f}s\n"
            f"  Content: {clip.get('clip_text', 'N/A')}\n"
            f"  Relevance Score: {clip.get('score', 0):.3f}"
        )
    
    return "\n\n".join(formatted)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
