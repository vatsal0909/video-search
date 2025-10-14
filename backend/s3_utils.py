"""
S3 utilities for generating presigned URLs for private bucket access
"""
import boto3
import os
from typing import Optional
from urllib.parse import urlparse

def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key
    
    Args:
        s3_url: S3 URL in format s3://bucket/key or https://bucket.s3.region.amazonaws.com/key
    
    Returns:
        Tuple of (bucket_name, object_key)
    """
    if s3_url.startswith('s3://'):
        # Format: s3://bucket/key
        parts = s3_url.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return bucket, key
    elif 's3' in s3_url and 'amazonaws.com' in s3_url:
        # Format: https://bucket.s3.region.amazonaws.com/key
        parsed = urlparse(s3_url)
        bucket = parsed.netloc.split('.')[0]
        key = parsed.path.lstrip('/')
        return bucket, key
    else:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")


def generate_presigned_url(s3_url: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for an S3 object
    
    Args:
        s3_url: S3 URL (s3://bucket/key or https format)
        expiration: Time in seconds for the presigned URL to remain valid (default: 1 hour)
    
    Returns:
        Presigned URL string or None if generation fails
    """
    try:
        # Parse S3 URL
        bucket, key = parse_s3_url(s3_url)
        
        # Get AWS credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Create session with explicit credentials if provided
        session_kwargs = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            if aws_session_token:
                session_kwargs['aws_session_token'] = aws_session_token
        
        # Create boto3 session and S3 client
        session = boto3.Session(**session_kwargs)
        s3_client = session.client('s3')
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key
            },
            ExpiresIn=expiration
        )
        
        return presigned_url
    
    except Exception as e:
        print(f"Error generating presigned URL for {s3_url}: {e}")
        return None


def add_presigned_urls_to_results(results: list, expiration: int = 3600) -> list:
    """
    Add presigned URLs to search results for video_path fields
    
    Args:
        results: List of search result dictionaries
        expiration: Time in seconds for presigned URLs to remain valid
    
    Returns:
        Updated results with presigned_url field added
    """
    for result in results:
        if 'video_path' in result:
            presigned_url = generate_presigned_url(result['video_path'], expiration)
            if presigned_url:
                result['presigned_url'] = presigned_url
            else:
                # Fallback to original URL if presigned generation fails
                result['presigned_url'] = result['video_path']
    
    return results
