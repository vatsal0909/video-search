import boto3
import json
from twelvelabs import TwelveLabs
from twelvelabs.embed import TasksStatusResponse
from typing import List, Dict
import os
import time

from dotenv import load_dotenv

load_dotenv()

class BedrockClient:
    def __init__(self, region: str = "us-east-1"):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using Bedrock LLM (Claude or GPT OSS)"""
        try:
            prompt = f"""Based on the following video clips:

{context}

Question: {question}

Provide a detailed answer with specific timestamps where relevant."""

            response = self.bedrock_runtime.invoke_model(
                # modelId='openai.gpt-oss-120b-1:0',
                modelId='amazon.nova-pro-v1:0',
                body=json.dumps({
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_output_tokens": 500,
                    "temperature": 0.7
                })
            )
            
            result = json.loads(response['body'].read())
            return result['output'][0]['content'][0]['text']
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}"

# class TwelveLabsClient:
#     def __init__(self):
#         api_key = os.getenv('TWELVELABS_API_KEY')
#         if not api_key:
#             raise ValueError("TWELVELABS_API_KEY environment variable not set")
        
#         self.client = TwelveLabs(api_key=api_key)
    
#     def generate_video_embeddings(self, video_path: str) -> List[Dict]:
#         """Generate embeddings for video using Marengo"""
#         try:
#             # Create embedding task
#             task = self.client.embed.tasks.create(
#                 model_name="Marengo-retrieval-2.7",
#                 video_url=video_path
#             )
            
#             print(f"Created task: {task.id}")
            
#             # Wait for task completion
#             task = self._wait_for_task(task.id)
            
#             if task.status != "ready":
#                 print(f"Task failed with status: {task.status}")
#                 return []
            
#             # Retrieve embeddings
#             embeddings = []
#             task_result = self.client.embed.tasks.retrieve(task.id)
            
#             # Process video embeddings
#             for segment in task_result.video_embedding.segments:
#                 print(segment)
#                 embeddings.append({
#                     "start_offset_sec": segment.start_offset_sec,
#                     "end_offset_sec": segment.end_offset_sec,
#                     "embedding_scope": segment.embedding_scope,
#                     "embedding": segment.float_
#                 })
            
#             return embeddings
        
#         except Exception as e:
#             print(f"Error generating video embeddings: {e}")
#             return []
    
#     def generate_text_embedding(self, text: str) -> List[float]:
#         """Generate embedding for text query using Marengo"""
#         try:
#             result = self.client.embed.create(
#                 model_name="Marengo-retrieval-2.7",
#                 text=text,
#                 text_truncate="none"
#             )
            
#             print(result)

#             if result.text_embedding and result.text_embedding.segments:
#                 return result.text_embedding.segments[0].float_
            
#             return []
        
#         except Exception as e:
#             print(f"Error generating text embedding: {e}")
#             return []
    
#     def _wait_for_task(self, task_id: str, max_wait: int = 600) -> TasksStatusResponse:
#         """Poll task until completion"""
#         start_time = time.time()
        
#         while time.time() - start_time < max_wait:
#             task = self.client.embed.tasks.retrieve(task_id)
            
#             if task.status in ["ready", "failed"]:
#                 return task
            
#             print(f"Task status: {task.status}")
#             time.sleep(10)
        
#         raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")


class BedrockTwelveLabsClient:
    """Client for generating embeddings using Marengo model via AWS Bedrock"""
    
    def __init__(self):
        # Get AWS credentials from environment variables
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
        
        # Create boto3 session
        session = boto3.Session(**session_kwargs)
        
        # Initialize clients with session
        self.bedrock_runtime = session.client('bedrock-runtime')
        self.bedrock = session.client('bedrock')
        self.region = region
        self.model_id = "twelvelabs.marengo-embed-2-7-v1:0"
        self.text_embed_model = "us.twelvelabs.marengo-embed-2-7-v1:0" # used for on-demand inference
        print(f"Initialized BedrockTwelveLabsClient in region: {self.region}")
    
    def generate_video_embeddings(self, video_uri: str) -> List[Dict]:
        """
        Generate embeddings for video using Marengo via AWS Bedrock.
        Uses async invocation for videos larger than 5MB.
        """
        try:
            print(f"Creating async embedding task for video: {video_uri}")
            
            # Get S3 output location from environment
            output_bucket = os.getenv("AWS_OUTPUT_BUCKET", "condenast-landingzone-useast1-943143228843-dev")
            output_prefix = os.getenv("AWS_OUTPUT_PREFIX", "bedrock-outputs/")
            
            # Create embedding request for Bedrock
            request_body = {
                "inputType": "video",
                "mediaSource": {
                    "s3Location": {
                        "uri": video_uri,
                        "bucketOwner": os.getenv("AWS_BUCKET_OWNER")
                    }
                }
            }
            
            # Start async invocation
            response = self.bedrock_runtime.start_async_invoke(
                modelId=self.model_id,
                modelInput=request_body,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"s3://{output_bucket}/{output_prefix}"
                    }
                }
            )
            
            invocation_arn = response['invocationArn']
            print(f"Started async invocation: {invocation_arn}")
            
            # Wait for completion and get results
            result = self._wait_for_async_invocation(invocation_arn)
            
            if not result:
                print("Failed to get results from async invocation")
                return []
            
            # Parse video embeddings from response
            embeddings = []
            if 'data' in result:
                for segment in result['data']:
                    print(f"Segment: {segment.get('startSec', 0)}-{segment.get('endSec', 0)}s")
                    embeddings.append({
                        "start_offset_sec": segment.get('startSec', 0),
                        "end_offset_sec": segment.get('endSec', 0),
                        "embedding_scope": segment.get('embeddingOption', 'clip'),
                        "embedding": segment.get('embedding', [])
                    })
            
            print(f"Generated {len(embeddings)} video embeddings")
            return embeddings
        
        except Exception as e:
            print(f"Error generating video embeddings: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text query using Marengo via AWS Bedrock"""
        try:
            request_body = {
                "inputType": "text",
                "inputText": text,
                "textTruncate": "none"
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.text_embed_model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            
            print(f"Text embedding response keys: {result.keys()}")
            
            # Extract text embedding
            if 'data' in result:
                segments = result['data']
                if segments and len(segments) > 0:
                    return segments[0].get('embedding', [])
            
            return []
        
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return []
    
    def _wait_for_async_invocation(self, invocation_arn: str, max_wait: int = 600) -> Dict:
        """
        Poll async invocation until completion and retrieve results from S3.
        """
        start_time = time.time()
        s3_client = boto3.client('s3', region_name=self.region)
        
        while time.time() - start_time < max_wait:
            try:
                # Get invocation status
                response = self.bedrock_runtime.get_async_invoke(
                    invocationArn=invocation_arn
                )
                
                status = response.get('status', 'UNKNOWN')
                print(f"Async invocation status: {status}")
                
                if status == 'Completed':
                    # Get output location from response
                    output_location = response.get('outputDataConfig', {}).get('s3OutputDataConfig', {}).get('s3Uri')
                    
                    if not output_location:
                        print("No output location found in response")
                        return {}
                    
                    print(f"Output location: {output_location}")
                    
                    # Parse S3 URI
                    # Format: s3://bucket/prefix/invocation-id/output.json
                    s3_parts = output_location.replace('s3://', '').split('/', 1)
                    bucket = s3_parts[0]
                    key = s3_parts[1] if len(s3_parts) > 1 else ''
                    
                    # Find the actual output file
                    # The output is typically at: prefix/invocation-id/output.json
                    invocation_id = invocation_arn.split('/')[-1]
                    possible_keys = [
                        # f"{key}/{invocation_id}/output.json",
                        f"{key}/output.json",
                        key
                    ]
                    
                    for output_key in possible_keys:
                        try:
                            print(f"Trying to read from s3://{bucket}/{output_key}")
                            obj = s3_client.get_object(Bucket=bucket, Key=output_key)
                            result = json.loads(obj['Body'].read().decode('utf-8'))
                            print(f"Successfully read results from S3")
                            return result
                        except s3_client.exceptions.NoSuchKey:
                            continue
                        except Exception as e:
                            print(f"Error reading from S3 key {output_key}: {e}")
                            continue
                    
                    print("Could not find output file in S3")
                    return {}
                
                elif status in ['Failed', 'Stopped']:
                    print(f"Async invocation failed with status: {status}")
                    failure_message = response.get('failureMessage', 'Unknown error')
                    print(f"Failure message: {failure_message}")
                    return {}
                
                # Still in progress
                time.sleep(10)
            
            except Exception as e:
                print(f"Error checking async invocation status: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)
        
        raise TimeoutError(f"Async invocation {invocation_arn} did not complete within {max_wait} seconds")

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from typing import List, Dict
import os

# class OpenSearchClient:
#     def __init__(self):
#         host = os.getenv('OPENSEARCH_HOST', 'localhost')
#         port = int(os.getenv('OPENSEARCH_PORT', 9200))
#         username = os.getenv('OPENSEARCH_USERNAME', 'admin')
#         password = os.getenv('OPENSEARCH_PASSWORD', 'Jod123@@123!')
        
#         self.client = OpenSearch(
#             hosts=[{'host': host, 'port': port}],
#             http_auth=(username, password),
#             http_compress=True,
#             use_ssl=True,
#             verify_certs=False,
#             ssl_show_warn=False
#         )
        
#         self._create_index_if_not_exists()
    
#     def _create_index_if_not_exists(self):
#         """Create video clips index with k-NN"""
#         index_name = "video_clips"
        
#         if not self.client.indices.exists(index=index_name):
#             index_body = {
#                 "settings": {
#                     "index": {
#                         "knn": True,
#                         "knn.algo_param.ef_search": 100
#                     }
#                 },
#                 "mappings": {
#                     "properties": {
#                         "video_id": {"type": "keyword"},
#                         "video_path": {"type": "keyword"},
#                         "timestamp_start": {"type": "float"},
#                         "timestamp_end": {"type": "float"},
#                         "clip_text": {"type": "text"},
#                         "embedding": {
#                             "type": "knn_vector",
#                             "dimension": 1024,
#                             "method": {
#                                 "name": "hnsw",
#                                 "space_type": "l2",
#                                 "engine": "lucene",
#                                 "parameters": {
#                                     "ef_construction": 128,
#                                     "m": 16
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
            
#             self.client.indices.create(index=index_name, body=index_body)
#             print(f"Created index: {index_name}")
    
#     def index_clip(self, video_id: str, video_path: str, 
#                    timestamp_start: float, timestamp_end: float,
#                    embedding: List[float], clip_text: str = ""):
#         """Index a video clip with embedding"""
#         doc = {
#             "video_id": video_id,
#             "video_path": video_path,
#             "timestamp_start": timestamp_start,
#             "timestamp_end": timestamp_end,
#             "clip_text": clip_text,
#             "embedding": embedding
#         }
        
#         response = self.client.index(
#             index="video_clips",
#             body=doc
#         )
#         return response
    
#     def search_similar_clips(self, query_embedding: List[float], 
#                             top_k: int = 10) -> List[Dict]:
#         """Search for similar video clips using k-NN"""
#         search_body = {
#             "size": top_k,
#             "query": {
#                 "knn": {
#                     "embedding": {
#                         "vector": query_embedding,
#                         "k": top_k
#                     }
#                 }
#             },
#             "_source": ["video_id", "video_path", "timestamp_start", 
#                        "timestamp_end", "clip_text"]
#         }
        
#         response = self.client.search(
#             index="video_clips",
#             body=search_body
#         )
        
#         results = []
#         for hit in response['hits']['hits']:
#             result = hit['_source']
#             result['score'] = hit['_score']
#             results.append(result)
        
#         return results
    
#     def hybrid_search(self, query_embedding: List[float], 
#                      text_query: str, top_k: int = 10) -> List[Dict]:
#         """Hybrid search combining vector and keyword search"""
#         search_body = {
#             "size": top_k,
#             "query": {
#                 "hybrid": {
#                     "queries": [
#                         {
#                             "knn": {
#                                 "embedding": {
#                                     "vector": query_embedding,
#                                     "k": top_k
#                                 }
#                             }
#                         },
#                         {
#                             "match": {
#                                 "clip_text": text_query
#                             }
#                         }
#                     ]
#                 }
#             }
#         }
        
#         response = self.client.search(
#             index="video_clips",
#             body=search_body
#         )
        
#         results = []
#         for hit in response['hits']['hits']:
#             result = hit['_source']
#             result['score'] = hit['_score']
#             results.append(result)
        
#         return results
    
#     def get_all_unique_videos(self) -> List[Dict]:
#         """Get all unique videos based on video_path with aggregated metadata"""
#         try:
#             # Use aggregations to get unique videos based on video_path
#             search_body = {
#                 "size": 0,
#                 "aggs": {
#                     "unique_videos": {
#                         "terms": {
#                             "field": "video_path",  # Group by video_path instead of video_id
#                             "size": 1000  # Adjust based on expected number of videos
#                         },
#                         "aggs": {
#                             "video_info": {
#                                 "top_hits": {
#                                     "size": 1,
#                                     "_source": ["video_id", "video_path", "timestamp_start", "timestamp_end"]
#                                 }
#                             },
#                             "total_clips": {
#                                 "value_count": {
#                                     "field": "video_path"
#                                 }
#                             },
#                             "max_timestamp": {
#                                 "max": {
#                                     "field": "timestamp_end"
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
            
#             response = self.client.search(
#                 index="video_clips",
#                 body=search_body
#             )
            
#             videos = []
#             for bucket in response['aggregations']['unique_videos']['buckets']:
#                 video_info = bucket['video_info']['hits']['hits'][0]['_source']
#                 max_duration = bucket['max_timestamp']['value']
#                 video_path = bucket['key']
                
#                 # Extract a readable name from the video path
#                 video_name = video_path.split('/')[-1].split('?')[0]  # Get filename from S3 URL
                
#                 videos.append({
#                     'video_id': video_info['video_id'],
#                     'video_path': video_path,
#                     'clips_count': bucket['doc_count'],
#                     'title': video_name or f"Video {video_info['video_id'][:8]}",
#                     'thumbnail_url': video_path,  # Use video URL for thumbnail generation
#                     'duration': max_duration if max_duration else None,
#                     'upload_date': None
#                 })
            
#             return videos
        
#         except Exception as e:
#             print(f"Error getting unique videos: {e}")
#             return []
    
#     def get_video_by_id(self, video_id: str) -> Dict:
#         """Get video details by video_id"""
#         try:
#             search_body = {
#                 "size": 1,
#                 "query": {
#                     "term": {
#                         "video_id": video_id
#                     }
#                 },
#                 "aggs": {
#                     "clips_count": {
#                         "value_count": {
#                             "field": "video_id"
#                         }
#                     }
#                 }
#             }
            
#             response = self.client.search(
#                 index="video_clips",
#                 body=search_body
#             )
            
#             if response['hits']['total']['value'] == 0:
#                 return None
            
#             video_info = response['hits']['hits'][0]['_source']
#             clips_count = response['aggregations']['clips_count']['value']
            
#             return {
#                 'video_id': video_info['video_id'],
#                 'video_path': video_info['video_path'],
#                 'clips_count': clips_count,
#                 'title': f"Video {video_id[:8]}",
#                 'thumbnail_url': None,
#                 'duration': None,
#                 'upload_date': None
#             }
        
#         except Exception as e:
#             print(f"Error getting video by ID: {e}")
#             return None


class BedrockOpensearchClient:
    """Client for AWS OpenSearch Serverless with AWS authentication"""
    
    def __init__(self):
        # Get AWS credentials from environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Get OpenSearch Serverless endpoint
        opensearch_host = os.getenv('OPENSEARCH_SERVERLESS_HOST')
        if not opensearch_host:
            raise ValueError("OPENSEARCH_SERVERLESS_HOST environment variable not set")
        
        # Strip protocol if accidentally included
        opensearch_host = opensearch_host.replace('https://', '').replace('http://', '').strip()
        
        # Create session with explicit credentials if provided
        session_kwargs = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            if aws_session_token:
                session_kwargs['aws_session_token'] = aws_session_token
        
        # Create boto3 session
        session = boto3.Session(**session_kwargs)
        credentials = session.get_credentials()

        print("Opensearch Session initialized")
        
        # Create AWS4Auth for OpenSearch Serverless
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'aoss',  # Service name for OpenSearch Serverless
            session_token=credentials.token
        )
        
        # Initialize OpenSearch client with AWS authentication
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )
        
        self.region = region
        print(f"Initialized BedrockOpensearchClient in region: {self.region}")
        
        self._create_index_if_not_exists()

        print("Index initialized")

        self.pipeline_exists = self._create_hybrid_search_pipeline()
        
        print("Hybrid search pipeline initialized")


    def _create_hybrid_search_pipeline(self):
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
                                "weights": [0.7, 0.3]  # 70% vector, 30% text
                            }
                        }
                    }
                }
            ]
        }
        
        try:
            # Check if pipeline exists
            try:
                self.client.search_pipeline.get(id="hybrid-norm-pipeline")
                print("Hybrid search pipeline already exists")
            except:
                # Create new pipeline
                self.client.search_pipeline.put(
                    id="hybrid-norm-pipeline",
                    body=pipeline_body
                )
                print("✓ Created hybrid search pipeline with min-max normalization")
        
        except Exception as e:
            print(f"✗ Pipeline creation error: {e}")
            # Fallback: pipeline not supported, will use manual normalization
            return False
        
        return True

    def _create_index_if_not_exists(self):
        """Create video clips index with k-NN for OpenSearch Serverless"""
        index_name = "video_clips"
        
        if not self.client.indices.exists(index=index_name):
            # OpenSearch Serverless uses nmslib or faiss engines, not lucene
            index_body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "video_id": {"type": "keyword"},
                        "video_path": {"type": "keyword"},
                        "timestamp_start": {"type": "float"},
                        "timestamp_end": {"type": "float"},
                        "clip_text": {"type": "text"},
                        "embedding_scope": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1024,
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "faiss",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                }
                            }
                        }
                    }
                }
            }
            
            self.client.indices.create(index=index_name, body=index_body)
            print(f"Created index: {index_name}")
    
    def index_clip(self, video_id: str, video_path: str, 
                   timestamp_start: float, timestamp_end: float,
                   embedding_scope: str, embedding: List[float], clip_text: str = ""):
        """Index a video clip with embedding"""
        doc = {
            "video_id": video_id,
            "video_path": video_path,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "clip_text": clip_text,
            "embedding_scope": embedding_scope,
            "embedding": embedding
        }
        
        response = self.client.index(
            index="video_clips",
            body=doc
        )
        return response
    
    def search_similar_clips(self, query_embedding: List[float], 
                            top_k: int = 10) -> List[Dict]:
        """Search for similar video clips using k-NN"""
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                        "filter": {
                            "term": {
                                "embedding_scope": "visual-image"
                            }
                        }
                    }
                }
            },
            "_source": ["video_id", "video_path", "timestamp_start", 
                       "timestamp_end", "clip_text", "embedding_scope"]
        }
        
        response = self.client.search(
            index="video_clips",
            body=search_body
        )
        
        results = []
        for hit in response['hits']['hits']:
            result = hit['_source']
            result['score'] = hit['_score']
            results.append(result)
        
        return results
    
    # def hybrid_search(self, query_embedding: List[float], 
    #                  text_query: str, top_k: int = 10) -> List[Dict]:
    #     """Hybrid search combining vector and keyword search"""
    #     search_body = {
    #         "size": top_k,
    #         "query": {
    #             "hybrid": {
    #                 "queries": [
    #                     {
    #                         "knn": {
    #                             "embedding": {
    #                                 "vector": query_embedding,
    #                                 "k": top_k,
    #                                 "filter": {
    #                                     "term": {
    #                                         "embedding_scope": "visual-image"
    #                                     }
    #                                 }
    #                             }
    #                         }
    #                     },
    #                     {
    #                         "match": {
    #                             "clip_text": text_query
    #                         }
    #                     }
    #                 ]
    #             }
    #         }
    #     }
        
    #     response = self.client.search(
    #         index="video_clips",
    #         body=search_body
    #     )
        
    #     results = []
    #     for hit in response['hits']['hits']:
    #         result = hit['_source']
    #         result['score'] = hit['_score']
    #         results.append(result)
        
    #     return results

    def hybrid_search(self, query_embedding: List[float], query_text: str, top_k: int = 10):
        """
        Hybrid search combining vector similarity and text relevance
        
        Args:
            query_embedding: Query embedding vector from Bedrock
            query_text: Original text query for BM25 matching
            top_k: Number of results to return
        
        Returns:
            List of search results with normalized scores
        """
        
        search_body = {
                "size": top_k, ##### EXPLICITLY DONE, Top K are taken from below
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k * 2
                                    }
                                }
                            },
                            {
                                "match": {
                                    "clip_text": {
                                        "query": query_text,
                                        "boost": 1.0
                                    }
                                }
                            }
                        ]
                    }
                },
                "post_filter": {
                    "term": {
                        "embedding_scope": "visual-image"
                    }
                }
            }
        
        # Use pipeline if available
        if self.pipeline_exists:
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
            response = self.client.search(**search_params)

            with open('temp.json', 'w') as f:
                f.write(json.dumps(response))
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result['_id'] = hit['_id']
                results.append(result)
            
            print(f"Found {len(results)} results with hybrid search")
            return results
        
        except Exception as e:
            print(f"Hybrid search error: {e}")
            # Fallback to vector-only search
            return self.search_similar_clips(query_embedding, top_k)
    
    def get_all_unique_videos(self) -> List[Dict]:
        """Get all unique videos based on video_path with aggregated metadata"""
        try:
            # Use aggregations to get unique videos based on video_path
            search_body = {
                "size": 0,
                "aggs": {
                    "unique_videos": {
                        "terms": {
                            "field": "video_path",  # Group by video_path instead of video_id
                            "size": 1000  # Adjust based on expected number of videos
                        },
                        "aggs": {
                            "video_info": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": ["video_id", "video_path", "timestamp_start", "timestamp_end"]
                                }
                            },
                            "total_clips": {
                                "value_count": {
                                    "field": "video_path"
                                }
                            },
                            "max_timestamp": {
                                "max": {
                                    "field": "timestamp_end"
                                }
                            }
                        }
                    }
                }
            }
            
            response = self.client.search(
                index="video_clips",
                body=search_body
            )
            
            videos = []
            for bucket in response['aggregations']['unique_videos']['buckets']:
                video_info = bucket['video_info']['hits']['hits'][0]['_source']
                max_duration = bucket['max_timestamp']['value']
                video_path = bucket['key']
                
                # Extract a readable name from the video path
                video_name = video_path.split('/')[-1].split('?')[0]  # Get filename from S3 URL
                
                videos.append({
                    'video_id': video_info['video_id'],
                    'video_path': video_path,
                    'clips_count': bucket['doc_count'],
                    'title': video_name or f"Video {video_info['video_id'][:8]}",
                    'thumbnail_url': video_path,  # Use video URL for thumbnail generation
                    'duration': max_duration if max_duration else None,
                    'upload_date': None
                })
            
            return videos
        
        except Exception as e:
            print(f"Error getting unique videos: {e}")
            return []
    
    def get_video_by_id(self, video_id: str) -> Dict:
        """Get video details by video_id"""
        try:
            search_body = {
                "size": 1,
                "query": {
                    "term": {
                        "video_id": video_id
                    }
                },
                "aggs": {
                    "clips_count": {
                        "value_count": {
                            "field": "video_id"
                        }
                    }
                }
            }
            
            response = self.client.search(
                index="video_clips",
                body=search_body
            )
            
            if response['hits']['total']['value'] == 0:
                return None
            
            video_info = response['hits']['hits'][0]['_source']
            clips_count = response['aggregations']['clips_count']['value']
            
            return {
                'video_id': video_info['video_id'],
                'video_path': video_info['video_path'],
                'clips_count': clips_count,
                'title': f"Video {video_id[:8]}",
                'thumbnail_url': None,
                'duration': None,
                'upload_date': None
            }
        
        except Exception as e:
            print(f"Error getting video by ID: {e}")
            return None

