# import json
# import boto3
# import os
# import subprocess
# import tempfile
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# import uuid


# def lambda_handler(event, context):
#     """
#     Read embeddings from S3, generate thumbnails for unique clips, and index into OpenSearch Serverless
#     """
#     try:
#         # Extract parameters from Step Functions
#         output_s3_path = event['outputS3Path']
#         part = event['part']
#         original_video = event['originalVideo']
        
#         # Use video key as ID (more meaningful than UUID)
#         video_id = str(uuid.uuid4())
        
#         print(f"Processing embeddings for part {part}")
#         print(f"Video ID: {video_id}")
#         print(f"Output S3 path: {output_s3_path}")
        
#         # Initialize clients
#         s3_client = boto3.client('s3', region_name='us-east-1')
#         opensearch_client = get_opensearch_client()
        
#         # Parse S3 path
#         bucket, key = parse_s3_uri(output_s3_path)
        
#         # Download embeddings from S3
#         embeddings_data = download_embeddings_from_s3(s3_client, bucket, key)
        
#         if not embeddings_data:
#             raise ValueError("No embeddings data found in S3")

#         print(f"✓ Successfully downloaded embeddings for part {part}")
        
#         # Generate thumbnails for unique clips using streaming
#         thumbnails = generate_thumbnails_streaming(
#             s3_client,
#             original_video['bucket'],
#             original_video['key'],
#             embeddings_data,
#             bucket,
#             key  # Store in thumbnails/ folder under same prefix
#         )
        
#         print(f"✓ Generated {len(thumbnails)} unique thumbnails")
        
#         # Index embeddings to OpenSearch with validation
#         indexed_count = index_embeddings_to_opensearch(
#             opensearch_client,
#             embeddings_data,
#             thumbnails,
#             video_id,
#             original_video,
#             part
#         )
        
#         print(f"✓ Successfully indexed {indexed_count} embeddings for part {part}")
        
#         return {
#             'statusCode': 200,
#             'part': part,
#             'videoId': video_id,
#             'embeddingsIndexed': indexed_count,
#             'thumbnailsGenerated': len(thumbnails),
#             'message': 'Successfully stored embeddings in OpenSearch'
#         }
        
#     except Exception as e:
#         print(f"Error storing embeddings: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise


# def get_opensearch_client():
#     """Initialize OpenSearch Serverless client with AWS authentication"""
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
    
#     client = OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=awsauth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20
#     )
    
#     # Ensure index exists
#     create_index_if_not_exists(client)
    
#     return client


# def create_index_if_not_exists(client):
#     """Create video_clips index if it doesn't exist"""
#     index_name = "video_clips"
    
#     if not client.indices.exists(index=index_name):
#         index_body = {
#             "settings": {
#                 "index": {
#                     "knn": True
#                 }
#             },
#             "mappings": {
#                 "properties": {
#                     "video_id": {"type": "keyword"},
#                     "video_path": {"type": "keyword"},
#                     "part": {"type": "integer"},
#                     "segment_index": {"type": "integer"},
#                     "timestamp_start": {"type": "float"},
#                     "timestamp_end": {"type": "float"},
#                     "clip_text": {"type": "text"},
#                     "embedding_scope": {"type": "keyword"},
#                     "thumbnail_path": {"type": "keyword"},
#                     "embedding": {
#                         "type": "knn_vector",
#                         "dimension": 1024,
#                         "method": {
#                             "name": "hnsw",
#                             "space_type": "l2",
#                             "engine": "faiss",
#                             "parameters": {
#                                 "ef_construction": 128,
#                                 "m": 16
#                             }
#                         }
#                     }
#                 }
#             }
#         }
        
#         client.indices.create(index=index_name, body=index_body)
#         print(f"Created index: {index_name}")


# def parse_s3_uri(s3_uri: str) -> tuple:
#     """Parse S3 URI into bucket and key"""
#     s3_parts = s3_uri.replace('s3://', '').split('/', 1)
#     bucket = s3_parts[0]
#     key = s3_parts[1] if len(s3_parts) > 1 else ''
#     return bucket, key


# def download_embeddings_from_s3(s3_client, bucket: str, key_prefix: str) -> dict:
#     """Download embeddings from S3, handling Bedrock's output structure"""
#     possible_keys = [
#         f"{key_prefix}/output.json"
#     ]
    
#     for key in possible_keys:
#         try:
#             print(f"Trying to read from s3://{bucket}/{key}")
#             obj = s3_client.get_object(Bucket=bucket, Key=key)
#             result = json.loads(obj['Body'].read().decode('utf-8'))
#             print(f"Successfully read embeddings from S3")
#             return result
#         except s3_client.exceptions.NoSuchKey:
#             continue
#         except Exception as e:
#             print(f"Error reading from S3 key {key}: {e}")
#             continue
    
#     raise ValueError(f"Could not find embeddings output in S3 at {bucket}/{key_prefix}")


# def generate_presigned_url(s3_client, bucket: str, key: str, expiration: int = 3600) -> str:
#     """Generate presigned URL for S3 object"""
#     try:
#         url = s3_client.generate_presigned_url(
#             'get_object',
#             Params={'Bucket': bucket, 'Key': key},
#             ExpiresIn=expiration
#         )
#         return url
#     except Exception as e:
#         print(f"Error generating presigned URL: {e}")
#         raise


# def generate_thumbnails_streaming(s3_client, video_bucket, video_key, 
#                                  embeddings_data, thumbnail_bucket, embeddings_prefix):
#     """
#     Generate thumbnails using streaming from S3 (no video download required)
#     Uses presigned URL + ffmpeg HTTP streaming
    
#     Returns: dict mapping (start_time, end_time) -> thumbnail S3 URI
#     """
#     thumbnails = {}
#     unique_clips = {}
    
#     if 'data' not in embeddings_data:
#         return thumbnails
    
#     segments = embeddings_data['data']
    
#     # Step 1: Find unique clips based on timestamp
#     for idx, segment in enumerate(segments):
#         start_time = round(segment.get('startSec', 0), 2)
#         end_time = round(segment.get('endSec', 0), 2)
#         clip_key = (start_time, end_time)
        
#         # Store only first occurrence of each unique clip
#         if clip_key not in unique_clips:
#             unique_clips[clip_key] = {
#                 'index': idx,
#                 'start': start_time,
#                 'end': end_time,
#                 'scope': segment.get('embeddingOption', 'clip')
#             }
    
#     print(f"Found {len(unique_clips)} unique clips from {len(segments)} segments")
    
#     # Step 2: Generate presigned URL for video
#     try:
#         # Generate presigned URL valid for 1 hour
#         video_url = generate_presigned_url(s3_client, video_bucket, video_key, 3600)
#         print(f"Generated presigned URL for video streaming")
        
#         # Calculate thumbnail S3 prefix
#         if 'embeddings/' in embeddings_prefix:
#             thumbnail_prefix = embeddings_prefix.replace('embeddings/', 'thumbnails/')
#         else:
#             thumbnail_prefix = embeddings_prefix.replace('/output.json', '/thumbnails')
        
#         # Step 3: Generate thumbnails using streaming
#         with tempfile.TemporaryDirectory() as tmpdir:
#             generated_count = 0
            
#             for clip_key, clip_info in unique_clips.items():
#                 try:
#                     start_time = clip_info['start']
#                     end_time = clip_info['end']
                    
#                     # Generate thumbnail filename
#                     thumbnail_filename = f"clip_{int(start_time*100)}_{int(end_time*100)}.jpg"
#                     thumbnail_path = f"{tmpdir}/{thumbnail_filename}"
                    
#                     # Use ffmpeg to extract frame directly from streaming URL
#                     cmd = [
#                         'ffmpeg',
#                         '-ss', str(start_time),          # Seek to timestamp
#                         '-i', video_url,                 # Input: presigned S3 URL
#                         '-vframes', '1',                 # Extract 1 frame
#                         '-q:v', '2',                     # High quality
#                         '-vf', 'scale=320:180',          # Resize to thumbnail (optional)
#                         '-y',                            # Overwrite output
#                         thumbnail_path
#                     ]
                    
#                     result = subprocess.run(
#                         cmd,
#                         check=True,
#                         capture_output=True,
#                         timeout=30,  # Increased timeout for streaming
#                         text=True
#                     )
                    
#                     # Upload thumbnail to S3
#                     thumbnail_s3_key = f"{thumbnail_prefix}/{thumbnail_filename}"
#                     s3_client.upload_file(
#                         thumbnail_path,
#                         thumbnail_bucket,
#                         thumbnail_s3_key,
#                         ExtraArgs={'ContentType': 'image/jpeg'}
#                     )
                    
#                     thumbnail_uri = f"s3://{thumbnail_bucket}/{thumbnail_s3_key}"
#                     thumbnails[clip_key] = thumbnail_uri
                    
#                     generated_count += 1
#                     if generated_count % 5 == 0:
#                         print(f"Generated {generated_count}/{len(unique_clips)} thumbnails")
                    
#                 except subprocess.TimeoutExpired:
#                     print(f"⚠️ Timeout generating thumbnail for clip {clip_key}")
#                     thumbnails[clip_key] = None
#                 except subprocess.CalledProcessError as e:
#                     print(f"⚠️ FFmpeg error for clip {clip_key}: {e.stderr}")
#                     thumbnails[clip_key] = None
#                 except Exception as e:
#                     print(f"⚠️ Error generating thumbnail for clip {clip_key}: {e}")
#                     thumbnails[clip_key] = None
#                     continue
            
#             print(f"✓ Successfully generated {generated_count} thumbnails via streaming")
    
#     except Exception as e:
#         print(f"Error in streaming thumbnail generation: {e}")
#         import traceback
#         traceback.print_exc()
    
#     return thumbnails


# def validate_embedding(embedding, expected_dim=1024):
#     """Validate embedding dimensions and data"""
#     if not isinstance(embedding, list):
#         return False, f"Embedding is not a list: {type(embedding)}"
    
#     if len(embedding) != expected_dim:
#         return False, f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
    
#     # Check if all elements are numbers
#     for i, val in enumerate(embedding):
#         if not isinstance(val, (int, float)):
#             return False, f"Embedding contains non-numeric value at index {i}: {val}"
#         if val != val:  # Check for NaN
#             return False, f"Embedding contains NaN at index {i}"
    
#     return True, "Valid"


# def index_embeddings_to_opensearch(opensearch_client, embeddings_data: dict,
#                                    thumbnails: dict, video_id: str, 
#                                    original_video: dict, part: int) -> int:
#     """Index embeddings into OpenSearch with validation (includes thumbnails)"""
#     index_name = "video_clips"
#     indexed_count = 0
    
#     # Parse embeddings from Bedrock response format
#     if 'data' in embeddings_data:
#         segments = embeddings_data['data']
#     else:
#         segments = []
    
#     print(f"Found {len(segments)} segments to index")
    
#     for idx, segment in enumerate(segments):
#         try:
#             # Extract embedding
#             embedding = segment.get('embedding', [])
            
#             # Validate embedding BEFORE indexing
#             is_valid, validation_msg = validate_embedding(embedding)
            
#             if not is_valid:
#                 print(f"⚠️ Skipping segment {idx}: {validation_msg}")
#                 print(f"   First 5 values: {embedding[:5] if len(embedding) >= 5 else embedding}")
#                 continue
            
#             # Get clip timestamp key for thumbnail lookup
#             start_time = round(segment.get('startSec', 0), 2)
#             end_time = round(segment.get('endSec', 0), 2)
#             clip_key = (start_time, end_time)
            
#             # Prepare document
#             doc = {
#                 'video_id': video_id,
#                 'video_path': f"s3://{original_video['bucket']}/{original_video['key']}",
#                 'part': part,
#                 'segment_index': idx,
#                 'timestamp_start': float(segment.get('startSec', 0)),
#                 'timestamp_end': float(segment.get('endSec', 0)),
#                 'clip_text': original_video['key'].split('/')[-1],
#                 'embedding_scope': segment.get('embeddingOption', 'clip'),
#                 'embedding': embedding,
#                 'thumbnail_path': thumbnails.get(clip_key)  # Same thumbnail for audio/visual-image/visual-text
#             }
            
#             # Index document
#             response = opensearch_client.index(
#                 index=index_name,
#                 body=doc
#             )
            
#             indexed_count += 1
            
#             if indexed_count % 10 == 0:
#                 print(f"Indexed {indexed_count}/{len(segments)} segments")
            
#         except Exception as e:
#             print(f"Error indexing segment {idx}: {e}")
#             print(f"  Embedding length: {len(embedding) if 'embedding' in locals() else 'N/A'}")
#             print(f"  Segment keys: {segment.keys()}")
#             continue
    
#     return indexed_count

############################################################################################################## Serverless Collection w/o thumbnails
# import json
# import boto3
# import os
# import hashlib
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# import uuid


# def lambda_handler(event, context):
#     """
#     Read embeddings from S3 and index into OpenSearch Serverless
#     """
#     try:
#         # Extract parameters from Step Functions
#         output_s3_path = event['outputS3Path']
#         part = event['part']
#         original_video = event['originalVideo']
        
#         # Use video key as ID (more meaningful than UUID)
#         video_id = str(uuid.uuid4())
        
#         print(f"Processing embeddings for part {part}")
#         print(f"Video ID: {video_id}")
#         print(f"Output S3 path: {output_s3_path}")
        
#         # Initialize clients
#         s3_client = boto3.client('s3', region_name='us-east-1')
#         opensearch_client = get_opensearch_client()
        
#         # Parse S3 path
#         bucket, key = parse_s3_uri(output_s3_path)
        
#         # Download embeddings from S3
#         embeddings_data = download_embeddings_from_s3(s3_client, bucket, key)
        
#         if not embeddings_data:
#             raise ValueError("No embeddings data found in S3")

#         print(f"✓ Successfully downloaded embeddings for part {part}")
        
#         # Index embeddings to OpenSearch with validation
#         indexed_count = index_embeddings_to_opensearch(
#             opensearch_client,
#             embeddings_data,
#             video_id,
#             original_video,
#             part
#         )
        
#         print(f"✓ Successfully indexed {indexed_count} embeddings for part {part}")
        
#         return {
#             'statusCode': 200,
#             'part': part,
#             'videoId': video_id,
#             'embeddingsIndexed': indexed_count,
#             'message': 'Successfully stored embeddings in OpenSearch'
#         }
        
#     except Exception as e:
#         print(f"Error storing embeddings: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise


# def get_opensearch_client():
#     """Initialize OpenSearch Serverless client with AWS authentication"""
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
    
#     client = OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=awsauth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20
#     )
    
#     # Ensure index exists
#     create_index_if_not_exists(client)
    
#     return client


# def create_index_if_not_exists(client):
#     """Create video_clips index if it doesn't exist"""
#     index_name = "video_clips"
    
#     if not client.indices.exists(index=index_name):
#         index_body = {
#             "settings": {
#                 "index": {
#                     "knn": True
#                 }
#             },
#             "mappings": {
#                 "properties": {
#                     "video_id": {"type": "keyword"},
#                     "video_path": {"type": "keyword"},
#                     "clip_id": {"type": "keyword"},  # Unique ID for each clip (same for all modalities)
#                     "part": {"type": "integer"},
#                     "segment_index": {"type": "integer"},
#                     "timestamp_start": {"type": "float"},
#                     "timestamp_end": {"type": "float"},
#                     "clip_text": {"type": "text"},
#                     "embedding_scope": {"type": "keyword"},
#                     "embedding": {
#                         "type": "knn_vector",
#                         "dimension": 1024,
#                         "method": {
#                             "name": "hnsw",
#                             "space_type": "l2",
#                             "engine": "faiss",
#                             "parameters": {
#                                 "ef_construction": 128,
#                                 "m": 16
#                             }
#                         }
#                     }
#                 }
#             }
#         }
        
#         client.indices.create(index=index_name, body=index_body)
#         print(f"Created index: {index_name}")


# def generate_clip_id(video_id: str, start_time: float, end_time: float) -> str:
#     """
#     Generate deterministic clip_id based on video_id and timestamps
#     Same clip (different modalities) will have the same clip_id
    
#     Args:
#         video_id: Video identifier
#         start_time: Clip start timestamp
#         end_time: Clip end timestamp
    
#     Returns:
#         Deterministic clip_id (hash of video_id + timestamps)
#     """
#     # Create a deterministic string from video_id and timestamps
#     clip_string = f"{video_id}_{start_time:.2f}_{end_time:.2f}"
    
#     # Generate SHA256 hash and take first 16 characters for shorter ID
#     clip_hash = hashlib.sha256(clip_string.encode()).hexdigest()[:16]
    
#     return f"clip_{clip_hash}"


# def parse_s3_uri(s3_uri: str) -> tuple:
#     """Parse S3 URI into bucket and key"""
#     s3_parts = s3_uri.replace('s3://', '').split('/', 1)
#     bucket = s3_parts[0]
#     key = s3_parts[1] if len(s3_parts) > 1 else ''
#     return bucket, key


# def download_embeddings_from_s3(s3_client, bucket: str, key_prefix: str) -> dict:
#     """Download embeddings from S3, handling Bedrock's output structure"""
#     possible_keys = [
#         f"{key_prefix}/output.json"
#     ]
    
#     for key in possible_keys:
#         try:
#             print(f"Trying to read from s3://{bucket}/{key}")
#             obj = s3_client.get_object(Bucket=bucket, Key=key)
#             result = json.loads(obj['Body'].read().decode('utf-8'))
#             print(f"Successfully read embeddings from S3")
#             return result
#         except s3_client.exceptions.NoSuchKey:
#             continue
#         except Exception as e:
#             print(f"Error reading from S3 key {key}: {e}")
#             continue
    
#     raise ValueError(f"Could not find embeddings output in S3 at {bucket}/{key_prefix}")


# def validate_embedding(embedding, expected_dim=1024):
#     """Validate embedding dimensions and data"""
#     if not isinstance(embedding, list):
#         return False, f"Embedding is not a list: {type(embedding)}"
    
#     if len(embedding) != expected_dim:
#         return False, f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
    
#     # Check if all elements are numbers
#     for i, val in enumerate(embedding):
#         if not isinstance(val, (int, float)):
#             return False, f"Embedding contains non-numeric value at index {i}: {val}"
#         if val != val:  # Check for NaN
#             return False, f"Embedding contains NaN at index {i}"
    
#     return True, "Valid"


# def index_embeddings_to_opensearch(opensearch_client, embeddings_data: dict,
#                                    video_id: str, original_video: dict, part: int) -> int:
#     """Index embeddings into OpenSearch with validation (includes clip_id for collapse)"""
#     index_name = "video_clips"
#     indexed_count = 0
    
#     # Parse embeddings from Bedrock response format
#     if 'data' in embeddings_data:
#         segments = embeddings_data['data']
#     else:
#         segments = []
    
#     print(f"Found {len(segments)} segments to index")
    
#     for idx, segment in enumerate(segments):
#         try:
#             # Extract embedding
#             embedding = segment.get('embedding', [])
            
#             # Validate embedding BEFORE indexing
#             is_valid, validation_msg = validate_embedding(embedding)
            
#             if not is_valid:
#                 print(f"⚠️ Skipping segment {idx}: {validation_msg}")
#                 print(f"   First 5 values: {embedding[:5] if len(embedding) >= 5 else embedding}")
#                 continue
            
#             # Get clip timestamps
#             start_time = round(segment.get('startSec', 0), 2)
#             end_time = round(segment.get('endSec', 0), 2)
            
#             # Generate deterministic clip_id (same for all modalities of this clip)
#             clip_id = generate_clip_id(video_id, start_time, end_time)
            
#             # Prepare document with clip_id
#             doc = {
#                 'video_id': video_id,
#                 'video_path': f"s3://{original_video['bucket']}/{original_video['key']}",
#                 'clip_id': clip_id,  # Same for audio/visual-image/visual-text
#                 'part': part,
#                 'segment_index': idx,
#                 'timestamp_start': float(segment.get('startSec', 0)),
#                 'timestamp_end': float(segment.get('endSec', 0)),
#                 'clip_text': original_video['key'].split('/')[-1],
#                 'embedding_scope': segment.get('embeddingOption', 'clip'),
#                 'embedding': embedding
#             }
            
#             # Index document
#             response = opensearch_client.index(
#                 index=index_name,
#                 body=doc
#             )
            
#             indexed_count += 1
            
#             if indexed_count % 10 == 0:
#                 print(f"Indexed {indexed_count}/{len(segments)} segments")
            
#         except Exception as e:
#             print(f"Error indexing segment {idx}: {e}")
#             print(f"  Embedding length: {len(embedding) if 'embedding' in locals() else 'N/A'}")
#             print(f"  Segment keys: {segment.keys()}")
#             continue
    
#     return indexed_count

##################################################################################################################################   Index with all modalities w thumbnails

# import json
# import boto3
# import os
# import hashlib
# from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
# from typing import Dict, List, Optional
# import uuid
# from collections import defaultdict
# from datetime import datetime
# import subprocess
# import tempfile
# import shutil

# # Configuration
# THUMBNAIL_BUCKET = os.environ.get('THUMBNAIL_BUCKET', 'demo-processed-useast1-943143228843-dev')
# THUMBNAIL_PREFIX = 'thumbnails/'
# EMBEDDING_DIMENSIONS = 1024
# INDEX_NAME = 'video_clips_consolidated'


# def lambda_handler(event, context):
#     """
#     Read embeddings from S3 and index into OpenSearch Cluster
#     Consolidates all modalities per clip into single documents with thumbnails
#     """
#     try:
#         # Extract parameters from Step Functions
#         output_s3_path = event['outputS3Path']
#         part = event['part']
#         original_video = event['originalVideo']
        
#         # Use video key as ID
#         video_id = str(uuid.uuid4())
        
#         print(f"Processing embeddings for part {part}")
#         print(f"Video ID: {video_id}")
#         print(f"Output S3 path: {output_s3_path}")
        
#         # Initialize clients
#         s3_client = boto3.client('s3', region_name='us-east-1')
#         opensearch_client = get_opensearch_client()
        
#         # Parse S3 path
#         bucket, key = parse_s3_uri(output_s3_path)
        
#         # Download embeddings from S3
#         embeddings_data = download_embeddings_from_s3(s3_client, bucket, key)
        
#         if not embeddings_data:
#             raise ValueError("No embeddings data found in S3")

#         print(f"✓ Successfully downloaded embeddings for part {part}")
        
#         # Index embeddings to OpenSearch with consolidation and thumbnails
#         indexed_count = index_embeddings_to_opensearch_consolidated(
#             opensearch_client,
#             s3_client,
#             embeddings_data,
#             video_id,
#             original_video,
#             part
#         )
        
#         print(f"✓ Successfully indexed {indexed_count} consolidated clips for part {part}")
        
#         return {
#             'statusCode': 200,
#             'part': part,
#             'videoId': video_id,
#             'clipsIndexed': indexed_count,
#             'message': 'Successfully stored consolidated embeddings in OpenSearch'
#         }
        
#     except Exception as e:
#         print(f"Error storing embeddings: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise


# def get_opensearch_client():
#     """Initialize OpenSearch Cluster client with AWS authentication"""
#     opensearch_host = os.environ['OPENSEARCH_CLUSTER_HOST']
#     opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
#     session = boto3.Session()
#     credentials = session.get_credentials()
    
#     auth = AWSV4SignerAuth(credentials, 'us-east-1', 'es')
    
#     client = OpenSearch(
#         hosts=[{'host': opensearch_host, 'port': 443}],
#         http_auth=auth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         pool_maxsize=20,
#         timeout=30,
#         retry_on_timeout=True,
#         max_retries=3
#     )
    
#     # Ensure index exists (only on first call)
#     create_index_if_not_exists(client)
    
#     return client


# def create_index_if_not_exists(client):
#     """
#     Create production-grade consolidated video_clips index
#     Optimized for accuracy, storage efficiency, and multimodal search
#     """
#     index_name = INDEX_NAME
    
#     if not client.indices.exists(index=index_name):
#         index_body = {
#             "settings": {
#                 "index": {
#                     "knn": True,
#                     "knn.algo_param.ef_search": 512,
#                     "number_of_shards": 2,
#                     "number_of_replicas": 1,
#                     "refresh_interval": "5s",
#                     "codec": "best_compression"
#                 }
#             },
#             "mappings": {
#                 "properties": {
#                     # Marengo embedding fields
#                     "emb_vis_image": {
#                         "type": "knn_vector",
#                         "dimension": EMBEDDING_DIMENSIONS,
#                         "method": {
#                             "name": "hnsw",
#                             "space_type": "cosinesimil",
#                             "engine": "faiss",
#                             "parameters": {
#                                 "ef_construction": 512,
#                                 "m": 32
#                             }
#                         }
#                     },
#                     "emb_vis_text": {
#                         "type": "knn_vector",
#                         "dimension": EMBEDDING_DIMENSIONS,
#                         "method": {
#                             "name": "hnsw",
#                             "space_type": "cosinesimil",
#                             "engine": "faiss",
#                             "parameters": {
#                                 "ef_construction": 512,
#                                 "m": 32
#                             }
#                         }
#                     },
#                     "emb_audio": {
#                         "type": "knn_vector",
#                         "dimension": EMBEDDING_DIMENSIONS,
#                         "method": {
#                             "name": "hnsw",
#                             "space_type": "cosinesimil",
#                             "engine": "faiss",
#                             "parameters": {
#                                 "ef_construction": 512,
#                                 "m": 32
#                             }
#                         }
#                     }
#                 }
#             }
#         }
        
#         client.indices.create(index=index_name, body=index_body)
#         print(f"✓ Created production-grade consolidated index: {index_name}")


# def generate_clip_id(video_id: str, start_time: float, end_time: float) -> str:
#     """Generate deterministic clip_id based on video_id and timestamps"""
#     clip_string = f"{video_id}_{start_time:.2f}_{end_time:.2f}"
#     clip_hash = hashlib.sha256(clip_string.encode()).hexdigest()[:16]
#     return f"clip_{clip_hash}"


# def parse_s3_uri(s3_uri: str) -> tuple:
#     """Parse S3 URI into bucket and key"""
#     s3_parts = s3_uri.replace('s3://', '').split('/', 1)
#     bucket = s3_parts[0]
#     key = s3_parts[1] if len(s3_parts) > 1 else ''
#     return bucket, key


# def download_embeddings_from_s3(s3_client, bucket: str, key_prefix: str) -> dict:
#     """Download embeddings from S3, handling Bedrock's output structure"""
#     possible_keys = [
#         f"{key_prefix}/output.json"
#     ]
    
#     for key in possible_keys:
#         try:
#             print(f"Trying to read from s3://{bucket}/{key}")
#             obj = s3_client.get_object(Bucket=bucket, Key=key)
#             result = json.loads(obj['Body'].read().decode('utf-8'))
#             print(f"✓ Successfully read embeddings from S3")
#             return result
#         except s3_client.exceptions.NoSuchKey:
#             continue
#         except Exception as e:
#             print(f"Error reading from S3 key {key}: {e}")
#             continue
    
#     raise ValueError(f"Could not find embeddings output in S3 at {bucket}/{key_prefix}")


# def validate_embedding(embedding, expected_dim=EMBEDDING_DIMENSIONS):
#     """Validate embedding dimensions and data"""
#     if not isinstance(embedding, list):
#         return False, f"Embedding is not a list: {type(embedding)}"
    
#     if len(embedding) != expected_dim:
#         return False, f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
    
#     for i, val in enumerate(embedding):
#         if not isinstance(val, (int, float)):
#             return False, f"Embedding contains non-numeric value at index {i}: {val}"
#         if val != val:  # Check for NaN
#             return False, f"Embedding contains NaN at index {i}"
    
#     return True, "Valid"


# def map_embedding_scope_to_field(scope: str) -> str:
#     """Map Marengo embedding scope to OpenSearch field name"""
#     scope_mapping = {
#         'visual-image': 'emb_vis_image',
#         'visual-text': 'emb_vis_text',
#         'audio': 'emb_audio',
#         'visual_image': 'emb_vis_image',
#         'visual_text': 'emb_vis_text',
#     }
    
#     return scope_mapping.get(scope, None)


# def extract_frame_at_timestamp(video_path: str, timestamp: float, temp_dir: str) -> Optional[str]:
#     """Extract a single frame from video at specified timestamp using ffmpeg"""
#     try:
#         frame_output = os.path.join(temp_dir, 'thumbnail_frame.jpg')
        
#         # Use ffmpeg to extract frame
#         cmd = [
#             'ffmpeg',
#             '-ss', str(timestamp),
#             '-i', video_path,
#             '-vframes', '1',
#             '-vf', 'scale=640:360',
#             '-y',
#             frame_output
#         ]
        
#         print(f"Extracting frame at {timestamp}s using ffmpeg")
#         result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
#         if result.returncode == 0 and os.path.exists(frame_output):
#             print(f"✓ Extracted frame: {frame_output}")
#             return frame_output
#         else:
#             print(f"⚠️ ffmpeg failed: {result.stderr[:200]}")
#             return None
            
#     except FileNotFoundError:
#         print(f"✗ ffmpeg not found in Lambda environment")
#         return None
#     except subprocess.TimeoutExpired:
#         print(f"✗ ffmpeg timeout (video may be corrupted)")
#         return None
#     except Exception as e:
#         print(f"✗ Error extracting frame: {e}")
#         return None


# def upload_frame_to_s3(s3_client, frame_path: str) -> Optional[str]:
#     """Upload extracted frame to S3 and return S3 URI"""
#     try:
#         # Generate unique thumbnail filename
#         thumbnail_name = f"{uuid.uuid4()}.jpg"
#         thumbnail_key = f"{THUMBNAIL_PREFIX}{thumbnail_name}"
        
#         # Read frame file
#         with open(frame_path, 'rb') as f:
#             frame_data = f.read()
        
#         # Upload to S3
#         s3_client.put_object(
#             Bucket=THUMBNAIL_BUCKET,
#             Key=thumbnail_key,
#             Body=frame_data,
#             ContentType='image/jpeg'
#         )
        
#         # Return S3 URI
#         s3_uri = f"s3://{THUMBNAIL_BUCKET}/{thumbnail_key}"
#         print(f"✓ Uploaded thumbnail to {s3_uri}")
#         return s3_uri
        
#     except Exception as e:
#         print(f"✗ Error uploading frame: {str(e)[:100]}")
#         return None


# def index_embeddings_to_opensearch_consolidated(opensearch_client, s3_client, embeddings_data: dict,
#                                                 video_id: str, original_video: dict, part: int) -> int:
#     """
#     Index embeddings into OpenSearch with consolidation and thumbnail generation
#     """
#     index_name = INDEX_NAME
    
#     if 'data' in embeddings_data:
#         segments = embeddings_data['data']
#     else:
#         segments = []
    
#     print(f"Found {len(segments)} segments to consolidate")
    
#     # Extract video info
#     video_name = original_video['key'].split('/')[-1].replace('-', ' ').replace('_', ' ')
#     video_s3_uri = f"s3://{original_video['bucket']}/{original_video['key']}"
    
#     # Step 1: Group embeddings by clip_id
#     clips_by_id = defaultdict(lambda: {
#         'metadata': None,
#         'embeddings': {}
#     })
    
#     for idx, segment in enumerate(segments):
#         try:
#             embedding = segment.get('embedding', [])
            
#             # Validate embedding
#             is_valid, validation_msg = validate_embedding(embedding)
#             if not is_valid:
#                 print(f"⚠️ Skipping segment {idx}: {validation_msg}")
#                 continue
            
#             start_time = round(segment.get('startSec', 0), 2)
#             end_time = round(segment.get('endSec', 0), 2)
#             embedding_scope = segment.get('embeddingOption', 'unknown')
            
#             # Generate clip ID
#             clip_id = generate_clip_id(video_id, start_time, end_time)
            
#             # Store metadata (once per clip)
#             if clips_by_id[clip_id]['metadata'] is None:
#                 clips_by_id[clip_id]['metadata'] = {
#                     'video_id': video_id,
#                     'video_path': video_s3_uri,
#                     'video_name': video_name,
#                     'clip_id': clip_id,
#                     'part': part,
#                     'timestamp_start': float(start_time),
#                     'timestamp_end': float(end_time),
#                     'clip_duration': float(end_time - start_time),
#                     'clip_text': video_name,
#                     'created_at': datetime.utcnow().isoformat()
#                 }
            
#             # Map scope to field name and store embedding
#             field_name = map_embedding_scope_to_field(embedding_scope)
#             if field_name:
#                 clips_by_id[clip_id]['embeddings'][field_name] = embedding
#                 print(f"  Clip {clip_id[:8]}... - Added {embedding_scope} → {field_name}")
#             else:
#                 print(f"⚠️ Unknown embedding scope: {embedding_scope}")
            
#         except Exception as e:
#             print(f"Error processing segment {idx}: {e}")
#             continue
    
#     print(f"✓ Consolidated {len(segments)} segments into {len(clips_by_id)} unique clips")
    
#     # ===== NEW: Download video ONCE before processing clips =====
#     temp_dir = tempfile.mkdtemp()
#     video_path = None
    
#     try:
#         # Download video once for all clips
#         video_path = os.path.join(temp_dir, 'video.mp4')
        
#         try:
#             print(f"Downloading video from s3://{original_video['bucket']}/{original_video['key']}")
#             s3_client.download_file(original_video['bucket'], original_video['key'], video_path)
#             print(f"✓ Downloaded video to {video_path} (will reuse for all {len(clips_by_id)} clips)")
#         except Exception as e:
#             print(f"⚠️ Cannot download video: {str(e)[:100]}")
#             print(f"  Skipping thumbnail generation for all clips")
#             video_path = None
        
#         # Step 2: Index consolidated clips with thumbnails
#         indexed_count = 0
        
#         for clip_id, clip_data in clips_by_id.items():
#             try:
#                 # Build document
#                 doc = clip_data['metadata'].copy()
#                 doc.update(clip_data['embeddings'])
                
#                 # Skip if no embeddings
#                 if len(clip_data['embeddings']) == 0:
#                     print(f"⚠️ Skipping clip {clip_id} - no valid embeddings")
#                     continue
                
#                 # Generate thumbnail using the already-downloaded video
#                 if video_path and os.path.exists(video_path):
#                     thumbnail_uri = generate_thumbnail_from_downloaded_video(
#                         s3_client,
#                         video_path,
#                         doc['timestamp_start']
#                     )
                    
#                     if thumbnail_uri:
#                         doc['thumbnail_path'] = thumbnail_uri
#                         print(f"  ✓ Added thumbnail: {thumbnail_uri}")
#                     else:
#                         doc['thumbnail_path'] = None
#                         print(f"  ⚠️ No thumbnail generated")
#                 else:
#                     doc['thumbnail_path'] = None
#                     print(f"  ⚠️ Video not available, skipping thumbnail")
                
#                 # Index document
#                 response = opensearch_client.index(
#                     index=index_name,
#                     id=clip_id,
#                     body=doc
#                 )
                
#                 indexed_count += 1
                
#                 if indexed_count % 10 == 0:
#                     print(f"Indexed {indexed_count}/{len(clips_by_id)} consolidated clips")
                
#                 # Log details
#                 modalities = list(clip_data['embeddings'].keys())
#                 duration = doc['clip_duration']
#                 print(f"  ✓ Clip {clip_id[:8]}... indexed:")
#                 print(f"     Duration: {duration:.2f}s")
#                 print(f"     Modalities ({len(modalities)}): {modalities}")
                
#             except Exception as e:
#                 print(f"Error indexing clip {clip_id}: {e}")
#                 continue
        
#         print(f"✓ Successfully indexed {indexed_count} consolidated clips with thumbnails")
        
#         return indexed_count
        
#     finally:
#         # Cleanup: Delete temporary directory and video file
#         if os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)
#             print(f"✓ Cleaned up temporary directory")


# def generate_thumbnail_from_downloaded_video(s3_client, video_path: str, timestamp: float) -> Optional[str]:
#     """
#     Generate thumbnail from already-downloaded video at specified timestamp
#     Returns S3 URI of generated thumbnail
#     """
#     try:
#         print(f"Generating thumbnail at {timestamp}s from {video_path}")
        
#         # Create temporary directory for frame
#         temp_dir = tempfile.mkdtemp()
        
#         try:
#             # Extract frame at timestamp
#             frame_path = extract_frame_at_timestamp(video_path, timestamp, temp_dir)
            
#             if frame_path:
#                 # Upload frame to S3
#                 thumbnail_s3_uri = upload_frame_to_s3(s3_client, frame_path)
#                 print(f"✓ Generated thumbnail: {thumbnail_s3_uri}")
#                 return thumbnail_s3_uri
#             else:
#                 print(f"⚠️ Frame extraction failed")
#                 return None
                
#         finally:
#             # Cleanup frame temp directory
#             if os.path.exists(temp_dir):
#                 shutil.rmtree(temp_dir)
        
#     except Exception as e:
#         print(f"✗ Error generating thumbnail: {e}")
#         return None


############################################################################################################################################## Marengo 3.0 w thumbnails


import json
import boto3
import os
import hashlib
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import Dict, List, Optional
import uuid
from collections import defaultdict
from datetime import datetime
import subprocess
import tempfile
import shutil

# Configuration
THUMBNAIL_BUCKET = os.environ.get('THUMBNAIL_BUCKET', 'demo-processed-useast1-943143228843-dev')
THUMBNAIL_PREFIX = 'thumbnails/'
EMBEDDING_DIMENSIONS = 512
INDEX_NAME = 'video_clips_3_lucene'

def lambda_handler(event, context):
    """
    Read embeddings from S3 and index into OpenSearch Cluster
    Consolidates all modalities per clip into single documents with thumbnails
    """
    try:
        # Extract parameters from Step Functions
        output_s3_path = event['outputS3Path']
        part = event['part']
        original_video = event['originalVideo']
        
        # Use video key as ID
        video_id = str(uuid.uuid4())
        
        print(f"Processing embeddings for part {part}")
        print(f"Video ID: {video_id}")
        print(f"Output S3 path: {output_s3_path}")
        
        # Initialize clients
        s3_client = boto3.client('s3', region_name='us-east-1')
        opensearch_client = get_opensearch_client()
        
        # Parse S3 path
        bucket, key = parse_s3_uri(output_s3_path)
        
        # Download embeddings from S3
        embeddings_data = download_embeddings_from_s3(s3_client, bucket, key)
        
        if not embeddings_data:
            raise ValueError("No embeddings data found in S3")

        print(f"✓ Successfully downloaded embeddings for part {part}")
        
        # Index embeddings to OpenSearch with consolidation and thumbnails
        indexed_count = index_embeddings_to_opensearch_consolidated(
            opensearch_client,
            s3_client,
            embeddings_data,
            video_id,
            original_video,
            part
        )
        
        print(f"✓ Successfully indexed {indexed_count} consolidated clips for part {part}")
        
        return {
            'statusCode': 200,
            'part': part,
            'videoId': video_id,
            'clipsIndexed': indexed_count,
            'message': 'Successfully stored consolidated embeddings in OpenSearch'
        }
        
    except Exception as e:
        print(f"Error storing embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def get_opensearch_client():
    """Initialize OpenSearch Cluster client with AWS authentication"""
    opensearch_host = os.environ['OPENSEARCH_CLUSTER_HOST']
    opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
    
    session = boto3.Session()
    credentials = session.get_credentials()
    
    auth = AWSV4SignerAuth(credentials, 'us-east-1', 'es')
    
    client = OpenSearch(
        hosts=[{'host': opensearch_host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=30,
        retry_on_timeout=True,
        max_retries=3
    )
    
    # Ensure index exists (only on first call)
    create_index_if_not_exists(client)
    
    return client


def create_index_if_not_exists(client):
    """
    Create production-grade consolidated video_clips index
    Optimized for accuracy, storage efficiency, and multimodal search
    """
    index_name = INDEX_NAME
    
    if not client.indices.exists(index=index_name):
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512,
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "refresh_interval": "5s"
                }
            },
            "mappings": {
                "properties": {
                    # Marengo embedding fields
                    "emb_visual": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSIONS,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 32
                            }
                        }
                    },
                    "emb_transcription": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSIONS,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 32
                            }
                        }
                    },
                    "emb_audio": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSIONS,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 32
                            }
                        }
                    }
                }
            }
        }
        
        client.indices.create(index=index_name, body=index_body)
        print(f"✓ Created production-grade consolidated index: {index_name}")


def generate_clip_id(video_id: str, start_time: float, end_time: float) -> str:
    """Generate deterministic clip_id based on video_id and timestamps"""
    clip_string = f"{video_id}_{start_time:.2f}_{end_time:.2f}"
    clip_hash = hashlib.sha256(clip_string.encode()).hexdigest()[:16]
    return f"clip_{clip_hash}"


def parse_s3_uri(s3_uri: str) -> tuple:
    """Parse S3 URI into bucket and key"""
    s3_parts = s3_uri.replace('s3://', '').split('/', 1)
    bucket = s3_parts[0]
    key = s3_parts[1] if len(s3_parts) > 1 else ''
    return bucket, key


def download_embeddings_from_s3(s3_client, bucket: str, key_prefix: str) -> dict:
    """Download embeddings from S3, handling Bedrock's output structure"""
    possible_keys = [
        f"{key_prefix}/output.json"
    ]
    
    for key in possible_keys:
        try:
            print(f"Trying to read from s3://{bucket}/{key}")
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            result = json.loads(obj['Body'].read().decode('utf-8'))
            print(f"✓ Successfully read embeddings from S3")
            return result
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception as e:
            print(f"Error reading from S3 key {key}: {e}")
            continue
    
    raise ValueError(f"Could not find embeddings output in S3 at {bucket}/{key_prefix}")


def validate_embedding(embedding, expected_dim=EMBEDDING_DIMENSIONS):
    """Validate embedding dimensions and data"""
    if not isinstance(embedding, list):
        return False, f"Embedding is not a list: {type(embedding)}"
    
    if len(embedding) != expected_dim:
        return False, f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
    
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            return False, f"Embedding contains non-numeric value at index {i}: {val}"
        if val != val:  # Check for NaN
            return False, f"Embedding contains NaN at index {i}"
    
    return True, "Valid"


def map_embedding_scope_to_field(scope: str) -> str:
    """Map Marengo embedding scope to OpenSearch field name"""
    scope_mapping = {
        'visual': 'emb_visual',
        'audio': 'emb_audio',
        'transcription': 'emb_transcription'
    }
    
    return scope_mapping.get(scope, None)


def extract_frame_at_timestamp(video_path: str, timestamp: float, temp_dir: str) -> Optional[str]:
    """Extract a single frame from video at specified timestamp using ffmpeg"""
    try:
        frame_output = os.path.join(temp_dir, 'thumbnail_frame.jpg')
        
        # Use ffmpeg to extract frame
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-vf', 'scale=640:360',
            '-y',
            frame_output
        ]
        
        print(f"Extracting frame at {timestamp}s using ffmpeg")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(frame_output):
            print(f"✓ Extracted frame: {frame_output}")
            return frame_output
        else:
            print(f"⚠️ ffmpeg failed: {result.stderr[:200]}")
            return None
            
    except FileNotFoundError:
        print(f"✗ ffmpeg not found in Lambda environment")
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ ffmpeg timeout (video may be corrupted)")
        return None
    except Exception as e:
        print(f"✗ Error extracting frame: {e}")
        return None


def upload_frame_to_s3(s3_client, frame_path: str) -> Optional[str]:
    """Upload extracted frame to S3 and return S3 URI"""
    try:
        # Generate unique thumbnail filename
        thumbnail_name = f"{uuid.uuid4()}.jpg"
        thumbnail_key = f"{THUMBNAIL_PREFIX}{thumbnail_name}"
        
        # Read frame file
        with open(frame_path, 'rb') as f:
            frame_data = f.read()
        
        # Upload to S3
        s3_client.put_object(
            Bucket=THUMBNAIL_BUCKET,
            Key=thumbnail_key,
            Body=frame_data,
            ContentType='image/jpeg'
        )
        
        # Return S3 URI
        s3_uri = f"s3://{THUMBNAIL_BUCKET}/{thumbnail_key}"
        print(f"✓ Uploaded thumbnail to {s3_uri}")
        return s3_uri
        
    except Exception as e:
        print(f"✗ Error uploading frame: {str(e)[:100]}")
        return None


def index_embeddings_to_opensearch_consolidated(opensearch_client, s3_client, embeddings_data: dict,
                                                video_id: str, original_video: dict, part: int) -> int:
    """
    Index embeddings into OpenSearch with consolidation and thumbnail generation
    """
    index_name = INDEX_NAME
    
    if 'data' in embeddings_data:
        segments = embeddings_data['data']
    else:
        segments = []
    
    print(f"Found {len(segments)} segments to consolidate")
    
    # Extract video info
    video_name = original_video['key'].split('/')[-1].replace('-', ' ').replace('_', ' ')
    video_s3_uri = f"s3://{original_video['bucket']}/{original_video['key']}"
    
    # Step 1: Group embeddings by clip_id
    clips_by_id = defaultdict(lambda: {
        'metadata': None,
        'embeddings': {}
    })

    video_duration = 0

    for segment in segments:
        if video_duration > 0 and segment.get('startSec') == 0:
            break
        clip_duration = segment.get('endSec', 0) - segment.get('startSec', 0)
        video_duration += clip_duration

    
    for idx, segment in enumerate(segments):
        try:
            embedding = segment.get('embedding', [])
            
            # Validate embedding
            is_valid, validation_msg = validate_embedding(embedding)
            if not is_valid:
                print(f"⚠️ Skipping segment {idx}: {validation_msg}")
                continue
            
            start_time = round(segment.get('startSec', 0), 2)
            end_time = round(segment.get('endSec', 0), 2)
            embedding_scope = segment.get('embeddingOption', 'unknown')
            
            # Generate clip ID
            clip_id = generate_clip_id(video_id, start_time, end_time)
            
            # Store metadata (once per clip)
            if clips_by_id[clip_id]['metadata'] is None:
                clips_by_id[clip_id]['metadata'] = {
                    'video_id': video_id,
                    'video_path': video_s3_uri,
                    'video_name': video_name,
                    'video_duration_sec': round(video_duration, 2),
                    'clip_id': clip_id,
                    'part': part,
                    'timestamp_start': float(start_time),
                    'timestamp_end': float(end_time),
                    'clip_duration': float(end_time - start_time),
                    'clip_text': video_name,
                    'created_at': datetime.utcnow().isoformat()
                }
            
            # Map scope to field name and store embedding
            field_name = map_embedding_scope_to_field(embedding_scope)
            if field_name:
                clips_by_id[clip_id]['embeddings'][field_name] = embedding
                print(f"  Clip {clip_id[:8]}... - Added {embedding_scope} → {field_name}")
            else:
                print(f"⚠️ Unknown embedding scope: {embedding_scope}")
            
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            continue
    
    print(f"✓ Consolidated {len(segments)} segments into {len(clips_by_id)} unique clips")
    
    # ===== NEW: Download video ONCE before processing clips =====
    temp_dir = tempfile.mkdtemp()
    video_path = None
    
    try:
        # Download video once for all clips
        video_path = os.path.join(temp_dir, 'video.mp4')
        
        try:
            print(f"Downloading video from s3://{original_video['bucket']}/{original_video['key']}")
            s3_client.download_file(original_video['bucket'], original_video['key'], video_path)
            print(f"✓ Downloaded video to {video_path} (will reuse for all {len(clips_by_id)} clips)")
        except Exception as e:
            print(f"⚠️ Cannot download video: {str(e)[:100]}")
            print(f"  Skipping thumbnail generation for all clips")
            video_path = None
        
        # Step 2: Index consolidated clips with thumbnails
        indexed_count = 0
        
        for clip_id, clip_data in clips_by_id.items():
            try:
                # Build document
                doc = clip_data['metadata'].copy()
                doc.update(clip_data['embeddings'])
                
                # Skip if no embeddings
                if len(clip_data['embeddings']) == 0:
                    print(f"⚠️ Skipping clip {clip_id} - no valid embeddings")
                    continue
                
                # Generate thumbnail using the already-downloaded video
                if video_path and os.path.exists(video_path):
                    thumbnail_uri = generate_thumbnail_from_downloaded_video(
                        s3_client,
                        video_path,
                        doc['timestamp_start']
                    )
                    
                    if thumbnail_uri:
                        doc['thumbnail_path'] = thumbnail_uri
                        print(f"  ✓ Added thumbnail: {thumbnail_uri}")
                    else:
                        doc['thumbnail_path'] = None
                        print(f"  ⚠️ No thumbnail generated")
                else:
                    doc['thumbnail_path'] = None
                    print(f"  ⚠️ Video not available, skipping thumbnail")
                
                # Index document
                response = opensearch_client.index(
                    index=index_name,
                    id=clip_id,
                    body=doc
                )
                
                indexed_count += 1
                
                if indexed_count % 10 == 0:
                    print(f"Indexed {indexed_count}/{len(clips_by_id)} consolidated clips")
                
                # Log details
                modalities = list(clip_data['embeddings'].keys())
                duration = doc['clip_duration']
                print(f"  ✓ Clip {clip_id[:8]}... indexed:")
                print(f"     Duration: {duration:.2f}s")
                print(f"     Modalities ({len(modalities)}): {modalities}")
                
            except Exception as e:
                print(f"Error indexing clip {clip_id}: {e}")
                continue
        
        print(f"✓ Successfully indexed {indexed_count} consolidated clips with thumbnails")
        
        return indexed_count
        
    finally:
        # Cleanup: Delete temporary directory and video file
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory")


def generate_thumbnail_from_downloaded_video(s3_client, video_path: str, timestamp: float) -> Optional[str]:
    """
    Generate thumbnail from already-downloaded video at specified timestamp
    Returns S3 URI of generated thumbnail
    """
    try:
        print(f"Generating thumbnail at {timestamp}s from {video_path}")
        
        # Create temporary directory for frame
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract frame at timestamp
            frame_path = extract_frame_at_timestamp(video_path, timestamp, temp_dir)
            
            if frame_path:
                # Upload frame to S3
                thumbnail_s3_uri = upload_frame_to_s3(s3_client, frame_path)
                print(f"✓ Generated thumbnail: {thumbnail_s3_uri}")
                return thumbnail_s3_uri
            else:
                print(f"⚠️ Frame extraction failed")
                return None
                
        finally:
            # Cleanup frame temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"✗ Error generating thumbnail: {e}")
        return None