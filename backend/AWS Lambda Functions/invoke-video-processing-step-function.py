import json
import boto3
import os
from datetime import datetime

from urllib.parse import unquote_plus

# Initialize Step Functions client
sfn_client = boto3.client('stepfunctions')

# Get state machine ARN from environment variable
STATE_MACHINE_ARN = os.environ['STATE_MACHINE_ARN']

def lambda_handler(event, context):
    """
    Triggered by S3 upload, starts Step Functions execution
    """
    
    # Parse S3 event
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        dst_bucket = 'demo-raw-useast1-943143228843-dev'
        timestamp = str(datetime.now().strftime('%m-%d-%Y_%H-%M-%S'))
        # Only process video files
        if not key.lower().endswith(('.mp4', '.mov', '.avi')):
            print(f"Skipping non-video file: {key}")
            continue
        
        # Prepare input for Step Functions
        sfn_input = {
            "detail": {
                "bucket": {
                    "name": bucket
                },
                "object": {
                    "key": key
                },
                "dst_bucket": {
                    "key": dst_bucket
                }
            }
        }
        
        # Start Step Functions execution
        try:
            response = sfn_client.start_execution(
                stateMachineArn=STATE_MACHINE_ARN,
                name=f"video-process-{key.replace('/', '-').replace('.', '-').replace(' ', '-')[:28]}-{timestamp}",
                input=json.dumps(sfn_input)
            )
            
            print(f"Started Step Functions execution: {response['executionArn']}")
            
        except Exception as e:
            print(f"Error starting Step Functions: {str(e)}")
            raise
    
    return {
        'statusCode': 200,
        'body': json.dumps('Step Functions triggered successfully')
    }
