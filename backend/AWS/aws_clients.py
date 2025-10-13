import boto3
import json
from typing import List, Dict
import os
import time
import base64
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from dotenv import load_dotenv

load_dotenv()


class AWSBedrockClient:
    """AWS Bedrock client for LLM generation"""
    
    def __init__(self, region: str = "us-east-1"):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.region = region
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using Bedrock LLM"""
        try:
            prompt = f"""Based on the following video clips:

{context}

Question: {question}

Provide a detailed answer with specific timestamps where relevant."""

            response = self.bedrock_runtime.invoke_model(
                modelId='openai.gpt-oss-120b-1:0',
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


class AWSMarengoClient:
    """AWS Bedrock client for generating embeddings using Marengo (TwelveLabs via Bedrock)"""
    
    def __init__(self, region: str = "us-east-1"):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.region = region
        # Marengo model ID in Bedrock
        self.model_id = "twelvelabs.marengo-retrieval-2.7"
    
    def generate_video_embeddings(self, video_url: str) -> List[Dict]:
        """
        Generate embeddings for video using Marengo via Bedrock.
        Uses async task pattern similar to TwelveLabs API.
        """
        try:
            print(f"Creating embedding task for video: {video_url}")
            
            # Create embedding task
            request_body = {
                "videoUrl": video_url,
                "modelId": self.model_id
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            
            # Parse video embeddings from response
            embeddings = []
            if 'videoEmbedding' in result and 'segments' in result['videoEmbedding']:
                for segment in result['videoEmbedding']['segments']:
                    embeddings.append({
                        "start_offset_sec": segment.get('startOffsetSec', 0),
                        "end_offset_sec": segment.get('endOffsetSec', 0),
                        "embedding_scope": segment.get('embeddingScope', 'clip'),
                        "embedding": segment.get('embedding', [])
                    })
            
            print(f"Generated {len(embeddings)} video embeddings")
            return embeddings
        
        except Exception as e:
            print(f"Error generating video embeddings: {e}")
            return []
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text query using Marengo via Bedrock"""
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
            
            # Extract text embedding
            if 'textEmbedding' in result and 'segments' in result['textEmbedding']:
                segments = result['textEmbedding']['segments']
                if segments and len(segments) > 0:
                    return segments[0].get('embedding', [])
            
            return []
        
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return []
    
    def _wait_for_task(self, task_id: str, max_wait: int = 600) -> Dict:
        """
        Poll task until completion (if async pattern is used).
        Note: Bedrock may handle this synchronously depending on implementation.
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check task status via Bedrock API
                response = self.bedrock.get_model_invocation_job(
                    jobIdentifier=task_id
                )
                
                status = response.get('status', 'UNKNOWN')
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    return response
                
                print(f"Task status: {status}")
                time.sleep(10)
            
            except Exception as e:
                print(f"Error checking task status: {e}")
                break
        
        raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")


class AWSOpenSearchClient:
    """AWS OpenSearch Service client with IAM authentication"""
    
    def __init__(self):
        # AWS OpenSearch Service endpoint
        host = os.getenv('AWS_OPENSEARCH_ENDPOINT', 'search-domain.us-east-1.es.amazonaws.com')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Get AWS credentials
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'es',
            session_token=credentials.token
        )
        
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
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
                        "knn.algo_param.ef_search": 100,
                        "number_of_shards": 2,
                        "number_of_replicas": 1
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
