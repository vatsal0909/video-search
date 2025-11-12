import json
import os
import time
from typing import Dict, List, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.exceptions import NotFoundError
import boto3
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class VideoEmbeddingsSegment:
    """Represents a single embedding segment within a video"""
    def __init__(
        self,
        embedding: List[float],
        start_sec: float,
        end_sec: float,
        embedding_option: str,
        thumbnail_path: Optional[str] = None
    ):
        self.embedding = embedding
        self.startSec = start_sec
        self.endSec = end_sec
        self.embeddingOption = embedding_option
        self.thumbnail_path = thumbnail_path
    
    def to_dict(self) -> Dict:
        return {
            "embedding": self.embedding,
            "startSec": self.startSec,
            "endSec": self.endSec,
            "embeddingOption": self.embeddingOption,
            "thumbnail_path": self.thumbnail_path
        }


class VideoEmbeddings:
    """Data model for a complete video with all its clip embeddings"""
    
    def __init__(
        self,
        video_id: str,
        video_name: str,
        s3_uri: str,
        key_frame_uri: Optional[str] = None,
        size_bytes: int = 0,
        duration_sec: float = 0.0,
        content_type: str = "video/mp4"
    ):
        self.video_id = video_id
        self.videoName = video_name
        self.s3URI = s3_uri
        self.keyFrameURI = key_frame_uri
        self.dataCreated = datetime.now(timezone.utc).isoformat()
        self.sizeBytes = size_bytes
        self.durationSec = duration_sec
        self.contentType = content_type
        self.embeddings: List[VideoEmbeddingsSegment] = []
    
    def add_segment(self, segment: VideoEmbeddingsSegment):
        """Add an embedding segment to this video"""
        self.embeddings.append(segment)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for OpenSearch indexing"""
        return {
            "videoName": self.videoName,
            "s3URI": self.s3URI,
            "keyFrameURI": self.keyFrameURI,
            "dataCreated": self.dataCreated,
            "sizeBytes": self.sizeBytes,
            "durationSec": self.durationSec,
            "contentType": self.contentType,
            "embeddings": [seg.to_dict() for seg in self.embeddings]
        }


class OpenSearchNestedIndexer:
    """Index multimodal embeddings into nested OpenSearch structure"""

    def __init__(self):
        """Initialize OpenSearch client with AWS credentials from .env"""
        self.opensearch_client = self._get_opensearch_client()
        self.source_index = "video_clips_consolidated"
        self.target_index = "video_clips_nested"


    def _get_opensearch_client(self):
        """Initialize OpenSearch Cluster client with AWS authentication from .env"""
        opensearch_host = os.getenv('OPENSEARCH_CLUSTER_HOST')
        if not opensearch_host:
            raise ValueError("OPENSEARCH_CLUSTER_HOST not found in .env")

        opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()

        # Read AWS credentials from .env
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # Optional
        region = os.getenv('AWS_REGION', 'us-east-1')

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in .env")

        logger.info(f"Connecting to OpenSearch at {opensearch_host} in region {region}")

        # Use boto3 session with explicit credentials
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=region
        )
        credentials = session.get_credentials()

        auth = AWSV4SignerAuth(credentials, region, 'es')

        client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
            timeout=120,
            connection_timeout=30
        )

        # Test connection
        try:
            info = client.info()
            logger.info(f"✓ Connected to OpenSearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {e}")
            raise

        return client


    def create_nested_index(self):
        """Create nested index with embeddings array structure"""
        if self.opensearch_client.indices.exists(index=self.target_index):
            logger.warning(f"Index {self.target_index} already exists. Attempting to delete...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.opensearch_client.indices.delete(index=self.target_index)
                    logger.info(f"✓ Successfully deleted existing index")
                    break
                except Exception as e:
                    if 'snapshot_in_progress' in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                            logger.warning(f"Snapshot in progress on index. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Could not delete index after {max_retries} attempts due to ongoing snapshot")
                            raise
                    else:
                        raise

        # Nested mapping with embeddings array
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "knn.algo_param.ef_search": 512,
                    "refresh_interval": "5s",
                    "codec": "best_compression"
                }
            },
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "nested",
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 1024,
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
                            "startSec": {"type": "float"},
                            "endSec": {"type": "float"},
                            "embeddingOption": {"type": "keyword"},
                            "thumbnail_path": {"type": "keyword"}
                        }
                    }
                }
            }
        }

        try:
            self.opensearch_client.indices.create(index=self.target_index, body=index_body)
            logger.info(f"✓ Created nested index: {self.target_index}")
            logger.info(f"  Structure:")
            logger.info(f"    - Top-level: Video metadata (videoName, s3URI, keyFrameURI, etc.)")
            logger.info(f"    - Nested: embeddings array with VideoEmbeddingsSegment objects")
            logger.info(f"    - Each segment: embedding (1024D), startSec, endSec, clipText, thumbnail_path")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise


    def read_and_consolidate_embeddings(self, batch_size: int = 1000) -> Dict[str, VideoEmbeddings]:
        """
        Read embeddings from local JSON file or OpenSearch source index
        Groups all clips from the same video into a single VideoEmbeddings document
        Saves raw JSON locally for inspection

        Args:
            batch_size: Number of documents to read in each scroll batch

        Returns:
            Dictionary of VideoEmbeddings objects indexed by video_id
        """
        json_filename = f"{self.source_index}.json"
        raw_documents = []

        # Check if local JSON file exists
        if os.path.exists(json_filename):
            logger.info(f"Found local JSON file: {json_filename}")
            try:
                with open(json_filename, 'r') as f:
                    raw_documents = json.load(f)
                logger.info(f"✓ Loaded {len(raw_documents)} documents from {json_filename}")
            except Exception as e:
                logger.error(f"Error loading local JSON file: {e}")
                raise
        else:
            logger.info(f"Reading embeddings from {self.source_index}...")
            
            try:
                # Use scroll API to read all documents
                query_body = {
                    "query": {"match_all": {}},
                    "size": batch_size
                }
            
                response = self.opensearch_client.search(
                    index=self.source_index,
                    body=query_body,
                    scroll='2m'
                )

                scroll_id = response['_scroll_id']

                while True:
                    hits = response['hits']['hits']

                    if not hits:
                        break

                    # Process each document
                    for hit in hits:
                        doc = hit['_source']
                        raw_documents.append(doc)

                    # Get next batch
                    response = self.opensearch_client.transport.perform_request(
                        'GET',
                        '/_search/scroll',
                        body={"scroll": "2m", "scroll_id": scroll_id}
                    )
                    scroll_id = response['_scroll_id']

                # Save raw JSON locally
                with open(json_filename, 'w') as f:
                    json.dump(raw_documents, f, indent=2)
                logger.info(f"✓ Saved raw documents to {json_filename}")

            except Exception as e:
                logger.error(f"Error reading embeddings from OpenSearch: {e}")
                raise

        # Consolidate documents by video_id
        video_embeddings: Dict[str, VideoEmbeddings] = {}
        total_read = 0
        segments_added = 0

        try:
            for doc in raw_documents:
                video_id = doc.get('video_id')
                clip_id = doc.get('clip_id')

                if not video_id:
                    logger.warning(f"Document {clip_id} has no video_id, skipping")
                    continue

                # Initialize VideoEmbeddings if first time seeing this video
                if video_id not in video_embeddings:
                    video_embeddings[video_id] = VideoEmbeddings(
                        video_id=video_id,
                        video_name=doc.get('video_name', 'Unknown'),
                        s3_uri=doc.get('video_path', ''),
                        key_frame_uri=doc.get('thumbnail_path'),
                        size_bytes=doc.get('size_bytes', 0),
                        duration_sec=doc.get('duration_sec', 0.0),
                        content_type=doc.get('content_type', 'video/mp4')
                    )

                # Create segments for each embedding type
                start_sec = doc.get('timestamp_start', 0.0)
                end_sec = doc.get('timestamp_end', 0.0)
                thumbnail_path = doc.get('thumbnail_path')
                
                # Process visual-text embedding
                emb_vis_text = doc.get('emb_vis_text', [])
                if emb_vis_text and isinstance(emb_vis_text, list) and len(emb_vis_text) > 0:
                    segment = VideoEmbeddingsSegment(
                        embedding=emb_vis_text,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        embedding_option='visual-text',
                        thumbnail_path=thumbnail_path
                    )
                    video_embeddings[video_id].add_segment(segment)
                    segments_added += 1
                else:
                    logger.warning(f"Document {clip_id} has empty/null emb_vis_text, skipping")
                
                # Process visual-image embedding
                emb_vis_image = doc.get('emb_vis_image', [])
                if emb_vis_image and isinstance(emb_vis_image, list) and len(emb_vis_image) > 0:
                    segment = VideoEmbeddingsSegment(
                        embedding=emb_vis_image,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        embedding_option='visual-image',
                        thumbnail_path=thumbnail_path
                    )
                    video_embeddings[video_id].add_segment(segment)
                    segments_added += 1
                else:
                    logger.warning(f"Document {clip_id} has empty/null emb_vis_image, skipping")
                
                # Process audio embedding
                emb_audio = doc.get('emb_audio', [])
                if emb_audio and isinstance(emb_audio, list) and len(emb_audio) > 0:
                    segment = VideoEmbeddingsSegment(
                        embedding=emb_audio,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        embedding_option='audio',
                        thumbnail_path=thumbnail_path
                    )
                    video_embeddings[video_id].add_segment(segment)
                    segments_added += 1
                else:
                    logger.warning(f"Document {clip_id} has empty/null emb_audio, skipping")
                
                total_read += 1

            logger.info(f"✓ Read {total_read} documents")
            logger.info(f"  Consolidated into {len(video_embeddings)} videos")
            logger.info(f"  Total segments added: {segments_added}")

            return video_embeddings

        except Exception as e:
            logger.error(f"Error consolidating embeddings: {e}")
            raise


    def index_video_embeddings(self, video_embeddings: Dict[str, VideoEmbeddings]):
        """
        Index VideoEmbeddings documents into nested index one by one

        Args:
            video_embeddings: Dictionary of VideoEmbeddings objects indexed by video_id
        """
        logger.info(f"Indexing {len(video_embeddings)} video embeddings documents...")

        indexed_count = 0

        try:
            for video_id, video_emb in video_embeddings.items():
                # Convert VideoEmbeddings to dictionary
                doc = video_emb.to_dict()

                try:
                    # Index document one by one
                    response = self.opensearch_client.index(
                        index=self.target_index,
                        id=video_id,
                        body=doc
                    )
                    
                    indexed_count += 1
                    logger.info(f"Indexed {indexed_count}/{len(video_embeddings)} videos - Video ID: {video_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to index video {video_id}: {e}")
                    raise

            logger.info(f"✓ Successfully indexed {indexed_count} video embeddings documents")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise


    def verify_indexing(self):
        """Verify nested indexing was successful"""
        try:
            # Count documents in both indexes
            source_count = self.opensearch_client.cat.count(index=self.source_index, format='json')
            target_count = self.opensearch_client.cat.count(index=self.target_index, format='json')

            source_docs = int(source_count[0]['count'])
            target_docs = int(target_count[0]['count'])

            logger.info(f"Source index ({self.source_index}): {source_docs} clip documents")
            logger.info(f"Target index ({self.target_index}): {target_docs} video documents (with nested embeddings)")

            # Sample a document
            sample = self.opensearch_client.search(
                index=self.target_index,
                body={"size": 1}
            )

            if sample['hits']['hits']:
                sample_doc = sample['hits']['hits'][0]['_source']
                logger.info(f"\nSample nested video document:")
                logger.info(f"  Video ID: {sample_doc.get('videoName')}")
                logger.info(f"  S3 URI: {sample_doc.get('s3URI')}")
                logger.info(f"  Key Frame URI: {sample_doc.get('keyFrameURI')}")
                logger.info(f"  Duration: {sample_doc.get('durationSec')}s")
                logger.info(f"  Size: {sample_doc.get('sizeBytes')} bytes")
                logger.info(f"  Created: {sample_doc.get('dataCreated')}")
                
                embeddings = sample_doc.get('embeddings', [])
                logger.info(f"  Embedded segments: {len(embeddings)}")
                
                if embeddings:
                    first_seg = embeddings[0]
                    logger.info(f"    First segment:")
                    logger.info(f"      - Time range: {first_seg.get('startSec')}s - {first_seg.get('endSec')}s")
                    logger.info(f"      - Embedding type: {first_seg.get('embeddingOption')}")
                    logger.info(f"      - Embedding dim: {len(first_seg.get('embedding', []))}D")
                    logger.info(f"      - Thumbnail: {first_seg.get('thumbnail_path')}")

        except Exception as e:
            logger.error(f"Error verifying indexing: {e}")


    def run_indexing(self):
        """Run the complete nested indexing workflow"""
        try:
            logger.info("=" * 70)
            
            logger.info("Starting OpenSearch Nested Embeddings Indexing")
            logger.info("=" * 70)
            logger.info("Structure:")
            logger.info("  - Top-level: Video metadata (videoName, s3URI, keyFrameURI, etc.)")
            logger.info("  - Nested: embeddings array with VideoEmbeddingsSegment objects")
            logger.info("  - Each segment: embedding (1024D), startSec, endSec, clipText, thumbnail_path")
            logger.info("=" * 70)

            # Step 1: Create nested index
            self.create_nested_index()

            # Step 2: Read and consolidate embeddings by video_id
            video_embeddings = self.read_and_consolidate_embeddings()

            # Step 3: Index video embeddings documents
            self.index_video_embeddings(video_embeddings)

            # Step 4: Verify indexing
            self.verify_indexing()

            logger.info("=" * 70)
            logger.info("✓ Nested indexing completed successfully!")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Nested indexing failed: {e}")
            raise


if __name__ == "__main__":
    try:
        # Create .env file if it doesn't exist (template)
        if not os.path.exists('.env'):
            logger.warning(".env file not found. Creating template...")
            env_template = """# OpenSearch Configuration
OPENSEARCH_CLUSTER_HOST=search-your-cluster.us-east-1.es.amazonaws.com

# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_SESSION_TOKEN=your_session_token_here  # Optional
AWS_REGION=us-east-1
"""
            with open('.env.template', 'w') as f:
                f.write(env_template)
            logger.info("Created .env.template - Please fill in your credentials and rename to .env")
            raise ValueError("Please configure .env file with your OpenSearch and AWS credentials")

        # Initialize and run nested indexer
        indexer = OpenSearchNestedIndexer()
        indexer.run_indexing()

    except KeyboardInterrupt:
        logger.info("\nNested indexing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise