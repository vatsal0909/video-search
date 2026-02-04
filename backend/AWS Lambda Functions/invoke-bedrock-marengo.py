import json
import boto3
import os

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    """Start Bedrock Marengo async job"""
    
    bucket = event['bucket']
    key = event['key']
    part = event['part']
    MARENGO_MODEL_VERSION = 3

    print(f"Processing {key} in bucket {bucket}")
    
    dst_bucket = 'demo-processed-useast1-943143228843-dev'

    if MARENGO_MODEL_VERSION == 3:
        modelId = 'twelvelabs.marengo-embed-3-0-v1:0'
        request_body = {
        "inputType": "video",
        "video": {
        "mediaSource": {
            "s3Location": {
                "uri": f"s3://{bucket}/{key}",
                "bucketOwner": os.environ["AWS_BUCKET_OWNER"]
                }
            },
        "embeddingOption": ["visual", "audio", "transcription"],
        "embeddingScope": ["clip"]
        }
    }
    else:
        modelId = "twelvelabs.marengo-embed-2-7-v1:0"
        request_body = {
        "inputType": "video",
        "mediaSource": {
            "s3Location": {
                "uri": f"s3://{bucket}/{key}",
                "bucketOwner": os.environ["AWS_BUCKET_OWNER"]
                }
            },
        "embeddingOption": ["visual-text", "audio", "visual-image"]
    }
    
    if MARENGO_MODEL_VERSION == 3:
        outputS3uri = f"s3://{dst_bucket}/embeddings-marengo-3/{key}/"
    else:
        outputS3uri = f"s3://{dst_bucket}/embeddings/{key}/"    
    
    # Start async model invocation job
    response = bedrock_runtime.start_async_invoke(
        modelId=modelId,
        modelInput=request_body,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": outputS3uri
            }
        }
    )
    
    print(f"Started Bedrock job: {response['invocationArn']} for part {part}")
    
    return {
        'jobId': response['invocationArn'],
        'invocationArn': response['invocationArn'],
        'part': part,
        'outputS3Uri': outputS3uri
    }
