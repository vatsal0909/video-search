"""
Video management routes for listing and retrieving video metadata
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from s3_utils import generate_presigned_url

router = APIRouter(prefix="/videos", tags=["videos"])

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

@router.get("/list", response_model=VideosListResponse)
async def list_all_videos():
    """
    Get all unique videos from the OpenSearch index
    Returns video metadata including S3 paths and clip counts
    """
    # Import here to avoid circular dependency
    from main import opensearch_client
    
    try:
        videos = opensearch_client.get_all_unique_videos()
        
        # Transform to response format
        video_list = []
        for video in videos:
            # Generate presigned URL for private S3 bucket access
            presigned_url = generate_presigned_url(video['video_path'], expiration=3600)
            
            video_list.append(VideoMetadata(
                video_id=video['video_id'],
                video_path=presigned_url if presigned_url else video['video_path'],
                title=video.get('title') or f"Video {video['video_id'][:8]}",
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
        raise HTTPException(status_code=500, detail=f"Error fetching videos: {str(e)}")

@router.get("/{video_id}", response_model=VideoMetadata)
async def get_video_details(video_id: str):
    """
    Get detailed information about a specific video
    """
    # Import here to avoid circular dependency
    from main import opensearch_client
    
    try:
        video = opensearch_client.get_video_by_id(video_id)
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Generate presigned URL for private S3 bucket access
        presigned_url = generate_presigned_url(video['video_path'], expiration=3600)
        
        return VideoMetadata(
            video_id=video['video_id'],
            video_path=presigned_url if presigned_url else video['video_path'],
            title=video.get('title'),
            thumbnail_url=video.get('thumbnail_url'),
            duration=video.get('duration'),
            upload_date=video.get('upload_date'),
            clips_count=video.get('clips_count', 0)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching video: {str(e)}")
