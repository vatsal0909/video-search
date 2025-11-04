

"""
OpenSearch Consolidation Script - Flat Structure
Reads embeddings from existing index and creates single documents with
separate fields for each modality based on Marengo model output:
- emb_vis_image (for visual-image scope)
- emb_vis_text (for visual-text scope)
- emb_audio (for audio scope)
"""

import json
import os
import time
from typing import Dict, List, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.exceptions import NotFoundError
import boto3
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class OpenSearchConsolidator:
    """Consolidate multimodal embeddings into single flat documents"""

    def __init__(self):
        """Initialize OpenSearch client with AWS credentials from .env"""
        self.opensearch_client = self._get_opensearch_client()
        self.source_index = "video_clips"
        self.target_index = "updated_video_clips_cosine_sim"


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


    def create_consolidated_index(self):
        """Create new consolidated index with separate embedding fields for Marengo modalities"""
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

        # New mapping with separate embedding fields for Marengo scopes
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                }
            },
            "mappings": {
                "properties": {
                    "video_id": {"type": "keyword"},
                    "video_path": {"type": "keyword"},
                    "clip_id": {"type": "keyword"},
                    "part": {"type": "integer"},
                    "timestamp_start": {"type": "float"},
                    "timestamp_end": {"type": "float"},
                    "clip_text": {"type": "text"},
                    # Marengo embedding modalities from AWS
                    "emb_vis_image": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 32
                            }
                        }
                    },
                    "emb_vis_text": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 32
                            }
                        }
                    },
                    "emb_audio": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 32
                            }
                        }
                    }
                }
            }
        }

        try:
            self.opensearch_client.indices.create(index=self.target_index, body=index_body)
            logger.info(f"✓ Created consolidated index: {self.target_index}")
            logger.info(f"  Marengo embedding fields:")
            logger.info(f"    - emb_vis_image (from visual-image scope)")
            logger.info(f"    - emb_vis_text (from visual-text scope)")
            logger.info(f"    - emb_audio (from audio scope)")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise


    def read_and_consolidate_embeddings(self, batch_size: int = 1000) -> Dict[str, Dict]:
        """
        Read embeddings from source index and consolidate by clip_id
        Maps embedding_scope values to flat field names

        Args:
            batch_size: Number of documents to read in each scroll batch

        Returns:
            Dictionary of consolidated documents indexed by clip_id
        """
        logger.info(f"Reading embeddings from {self.source_index}...")

        consolidated_docs = {}
        total_read = 0
        modality_counts = {}

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
                    clip_id = doc.get('clip_id')

                    if not clip_id:
                        logger.warning(f"Document {hit['_id']} has no clip_id, skipping")
                        continue

                    # Initialize consolidated doc if first time seeing this clip
                    if clip_id not in consolidated_docs:
                        consolidated_docs[clip_id] = {
                            'video_id': doc.get('video_id'),
                            'video_path': doc.get('video_path'),
                            'clip_id': clip_id,
                            'part': doc.get('part'),
                            'timestamp_start': doc.get('timestamp_start'),
                            'timestamp_end': doc.get('timestamp_end'),
                            'clip_text': doc.get('clip_text'),
                            # Separate fields for Marengo modalities
                            'emb_vis_image': None,
                            'emb_vis_text': None,
                            'emb_audio': None,
                        }

                    # Map Marengo embedding_scope to field name
                    embedding_scope = doc.get('embedding_scope', 'unknown')
                    embedding_vector = doc.get('embedding', [])

                    # EXACT mapping from Marengo model scopes
                    scope_to_field = {
                        'visual-image': 'emb_vis_image',
                        'visual-text': 'emb_vis_text',
                        'audio': 'emb_audio',
                    }

                    field_name = scope_to_field.get(embedding_scope)

                    if field_name:
                        # Store embedding in appropriate field (keep first one if duplicate)
                        if consolidated_docs[clip_id][field_name] is None:
                            consolidated_docs[clip_id][field_name] = embedding_vector
                            modality_counts[embedding_scope] = modality_counts.get(embedding_scope, 0) + 1
                    else:
                        logger.warning(f"Unknown embedding_scope: '{embedding_scope}'. Expected one of: 'visual-image', 'visual-text', 'audio'")

                    total_read += 1

                # Get next batch
                response = self.opensearch_client.transport.perform_request(
                    'GET',
                    '/_search/scroll',
                    body={"scroll": "2m", "scroll_id": scroll_id}
                )
                scroll_id = response['_scroll_id']

            logger.info(f"✓ Read {total_read} documents and consolidated into {len(consolidated_docs)} clips")
            logger.info(f"  Modality distribution:")
            for scope, count in sorted(modality_counts.items()):
                logger.info(f"    - {scope}: {count}")

            return consolidated_docs

        except Exception as e:
            logger.error(f"Error reading embeddings: {e}")
            raise


    def _bulk_index_with_retry(self, bulk_body: List, max_retries: int = 3):
        """
        Perform bulk indexing with exponential backoff retry logic
        
        Args:
            bulk_body: List of bulk operation documents
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response from bulk operation
        """
        from opensearchpy.exceptions import ConnectionTimeout
        
        for attempt in range(max_retries):
            try:
                response = self.opensearch_client.bulk(body=bulk_body)
                return response
            except (TimeoutError, ConnectionError, ConnectionTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Bulk operation timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Bulk operation failed after {max_retries} attempts")
                    raise

    def index_consolidated_documents(self, consolidated_docs: Dict[str, Dict], batch_size: int = 25):
        """
        Index consolidated documents into new index

        Args:
            consolidated_docs: Dictionary of consolidated documents
            batch_size: Number of documents to bulk index at once
        """
        logger.info(f"Indexing {len(consolidated_docs)} consolidated documents...")

        indexed_count = 0

        try:
            # Prepare bulk documents
            bulk_body = []

            for clip_id, doc in consolidated_docs.items():
                # Remove None values (modalities not present for this clip)
                doc_clean = {k: v for k, v in doc.items() if v is not None}

                # Add metadata for bulk indexing
                bulk_body.append({
                    "index": {
                        "_index": self.target_index,
                        "_id": clip_id
                    }
                })
                bulk_body.append(doc_clean)

                # Bulk index when batch size reached
                if len(bulk_body) >= batch_size * 2:  # *2 because each doc has metadata + body
                    response = self._bulk_index_with_retry(bulk_body)

                    if response.get('errors'):
                        logger.warning(f"Some documents failed to index")

                    indexed_count += batch_size
                    logger.info(f"Indexed {indexed_count}/{len(consolidated_docs)} documents")

                    bulk_body = []

            # Index remaining documents
            if bulk_body:
                response = self._bulk_index_with_retry(bulk_body)

                if response.get('errors'):
                    logger.warning(f"Some documents failed to index")

                indexed_count = len(consolidated_docs)

            logger.info(f"✓ Successfully indexed {indexed_count} consolidated documents")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise


    def verify_consolidation(self):
        """Verify consolidation was successful"""
        try:
            # Count documents in both indexes
            source_count = self.opensearch_client.cat.count(index=self.source_index, format='json')
            target_count = self.opensearch_client.cat.count(index=self.target_index, format='json')

            source_docs = int(source_count[0]['count'])
            target_docs = int(target_count[0]['count'])

            logger.info(f"Source index ({self.source_index}): {source_docs} documents")
            logger.info(f"Target index ({self.target_index}): {target_docs} consolidated documents")

            # Sample a document
            sample = self.opensearch_client.search(
                index=self.target_index,
                body={"size": 1}
            )

            if sample['hits']['hits']:
                sample_doc = sample['hits']['hits'][0]['_source']
                logger.info(f"\nSample consolidated document:")
                logger.info(f"  Clip ID: {sample_doc.get('clip_id')}")
                logger.info(f"  Video ID: {sample_doc.get('video_id')}")
                logger.info(f"  Timestamp: {sample_doc.get('timestamp_start')}s - {sample_doc.get('timestamp_end')}s")

                modalities = []
                for field in ['emb_vis_image', 'emb_vis_text', 'emb_audio']:
                    if field in sample_doc:
                        vector_dim = len(sample_doc.get(field, []))
                        original_scope = {
                            'emb_vis_image': 'visual-image',
                            'emb_vis_text': 'visual-text',
                            'emb_audio': 'audio'
                        }.get(field)
                        modalities.append(f"{field} ({vector_dim}D from {original_scope})")

                logger.info(f"  Available modalities:")
                for mod in modalities:
                    logger.info(f"    - {mod}")

        except Exception as e:
            logger.error(f"Error verifying consolidation: {e}")


    def run_consolidation(self):
        """Run the complete consolidation workflow"""
        try:
            logger.info("=" * 70)
            logger.info("Starting OpenSearch Consolidation (Marengo Modalities)")
            logger.info("=" * 70)
            logger.info("Scope mappings:")
            logger.info("  visual-image  → emb_vis_image")
            logger.info("  visual-text   → emb_vis_text")
            logger.info("  audio         → emb_audio")
            logger.info("=" * 70)

            # Step 1: Create consolidated index
            self.create_consolidated_index()

            # Step 2: Read and consolidate embeddings
            consolidated_docs = self.read_and_consolidate_embeddings()

            # Step 3: Index consolidated documents
            self.index_consolidated_documents(consolidated_docs)

            # Step 4: Verify consolidation
            self.verify_consolidation()

            logger.info("=" * 70)
            logger.info("✓ Consolidation completed successfully!")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            raise


if __name__ == "__main__":
    try:
        # Create .env file if it doesn't exist (template)
        if not os.path.exists('.env'):
            logger.warning(".env file not found. Creating template...")
            env_template = """
# OpenSearch Configuration
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

        # Initialize and run consolidator
        consolidator = OpenSearchConsolidator()
        consolidator.run_consolidation()

    except KeyboardInterrupt:
        logger.info("\nConsolidation interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise