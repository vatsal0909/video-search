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

class TwelveLabsClient:
    def __init__(self):
        api_key = os.getenv('TWELVELABS_API_KEY')
        if not api_key:
            raise ValueError("TWELVELABS_API_KEY environment variable not set")
        
        self.client = TwelveLabs(api_key=api_key)
    
    def generate_video_embeddings(self, video_path: str) -> List[Dict]:
        """Generate embeddings for video using Marengo"""
        try:
            # Create embedding task
            task = self.client.embed.tasks.create(
                model_name="Marengo-retrieval-2.7",
                video_url=video_path
            )
            
            print(f"Created task: {task.id}")
            
            # Wait for task completion
            task = self._wait_for_task(task.id)
            
            if task.status != "ready":
                print(f"Task failed with status: {task.status}")
                return []
            
            # Retrieve embeddings
            embeddings = []
            task_result = self.client.embed.tasks.retrieve(task.id)
            
            # Process video embeddings
            for segment in task_result.video_embedding.segments:
                print(segment)
                embeddings.append({
                    "start_offset_sec": segment.start_offset_sec,
                    "end_offset_sec": segment.end_offset_sec,
                    "embedding_scope": segment.embedding_scope,
                    "embedding": segment.float_
                })
            
            return embeddings
        
        except Exception as e:
            print(f"Error generating video embeddings: {e}")
            return []
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text query using Marengo"""
        try:
            result = self.client.embed.create(
                model_name="Marengo-retrieval-2.7",
                text=text,
                text_truncate="none"
            )
            
            print(result)

            if result.text_embedding and result.text_embedding.segments:
                return result.text_embedding.segments[0].float_
            
            return []
        
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return []
    
    def _wait_for_task(self, task_id: str, max_wait: int = 600) -> TasksStatusResponse:
        """Poll task until completion"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            task = self.client.embed.tasks.retrieve(task_id)
            
            if task.status in ["ready", "failed"]:
                return task
            
            print(f"Task status: {task.status}")
            time.sleep(10)
        
        raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")


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
        self.model_id = "twelvelabs.marengo-embed-2-7-v1:0" # Should use INFERENCE PROFILE ID instead of model id "us.twelvelabs.marengo-embed-2-7-v1:0"
        
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
                "text": text,
                "textTruncate": "none"
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            
            print(f"Text embedding response keys: {result.keys()}")
            
            # Extract text embedding
            if 'textEmbedding' in result and 'segments' in result['textEmbedding']:
                segments = result['textEmbedding']['segments']
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
                        f"{key}/{invocation_id}/output.json",
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

from opensearchpy import OpenSearch
from typing import List, Dict
import os

class OpenSearchClient:
    def __init__(self):
        host = os.getenv('OPENSEARCH_HOST', 'localhost')
        port = int(os.getenv('OPENSEARCH_PORT', 9200))
        username = os.getenv('OPENSEARCH_USERNAME', 'admin')
        password = os.getenv('OPENSEARCH_PASSWORD', 'Jod123@@123!')
        
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(username, password),
            http_compress=True,
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Create video clips index with k-NN"""
        index_name = "video_clips"
        
        if not self.client.indices.exists(index=index_name):
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "video_id": {"type": "keyword"},
                        "video_path": {"type": "keyword"},
                        "timestamp_start": {"type": "float"},
                        "timestamp_end": {"type": "float"},
                        "clip_text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1024,
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "lucene",
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
                   embedding: List[float], clip_text: str = ""):
        """Index a video clip with embedding"""
        doc = {
            "video_id": video_id,
            "video_path": video_path,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "clip_text": clip_text,
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
                        "k": top_k
                    }
                }
            },
            "_source": ["video_id", "video_path", "timestamp_start", 
                       "timestamp_end", "clip_text"]
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
    
    def hybrid_search(self, query_embedding: List[float], 
                     text_query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid search combining vector and keyword search"""
        search_body = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        },
                        {
                            "match": {
                                "clip_text": text_query
                            }
                        }
                    ]
                }
            }
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

