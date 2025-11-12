import boto3
import requests
import time
import json
import os
import logging
from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ---------------- Helper: SigV4-signed HTTP request ----------------
def signed_request(method, url, region, service, credentials, json_body=None, headers=None, timeout=60):
    """
    Perform an AWS SigV4 signed HTTP request using botocore + requests.
    """
    body = None
    if json_body is not None:
        body = json.dumps(json_body)
        if headers is None:
            headers = {}
        headers = dict(headers)
        headers.setdefault("Content-Type", "application/json")

    aws_request = AWSRequest(method=method.upper(), url=url, data=body, headers=headers or {})
    SigV4Auth(credentials, service, region).add_auth(aws_request)

    prepared = requests.Request(
        method=aws_request.method,
        url=aws_request.url,
        data=aws_request.body,
        headers=dict(aws_request.headers)
    ).prepare()

    session = requests.Session()
    response = session.send(prepared, timeout=timeout)
    return response


# ---------------- Lambda Handler ----------------
def lambda_handler(event, context):
    """
    AWS Lambda handler for creating OpenSearch snapshots.
    
    Environment variables required:
    - OPENSEARCH_CLUSTER_HOST: OpenSearch domain endpoint
    - SNAPSHOT_ROLE_ARN: IAM role ARN for snapshot permissions
    - S3_BUCKET: S3 bucket for snapshots (default: condenast-fe)
    - REPO_NAME: Repository name (default: condenast-snapshot-repo)
    """
    try:
        # ---------- CONFIG ----------
        region = "us-east-1"
        service = "es"

        domain = os.getenv(
            "OPENSEARCH_CLUSTER_HOST",
            "https://search-condenast-aos-domain-3hmon7me6ct3p5e46snecxe6f4.us-east-1.es.amazonaws.com"
        ).replace("https://", "").replace("http://", "")

        repo_name = os.getenv("REPO_NAME", "condenast-snapshot-repo")
        bucket = os.getenv("S3_BUCKET", "condenast-fe")
        role_arn = os.getenv("SNAPSHOT_ROLE_ARN", "arn:aws:iam::943143228843:role/condenast-opensearch-snapshot-role")
        snapshot_name = f"manual-snapshot-{int(time.time())}"

        if not domain:
            raise ValueError("OPENSEARCH_CLUSTER_HOST environment variable not set")
        if not role_arn:
            raise ValueError("SNAPSHOT_ROLE_ARN environment variable not set")

        logger.info(f"Config: domain={domain}, repo={repo_name}, bucket={bucket}")

        # ---------- AWS AUTH ----------
        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()

        headers = {"Content-Type": "application/json"}

        # ---------- REGISTER REPOSITORY ----------
        register_payload = {
            "type": "s3",
            "settings": {
                "bucket": bucket,
                "region": region,
                "role_arn": role_arn,
                "base_path": "opensearch-snapshots",
                "server_side_encryption": True
            }
        }

        url_repo = f"https://{domain}/_snapshot/{repo_name}"
        logger.info(f"Registering repo {repo_name} ...")

        r = signed_request("PUT", url_repo, region, service, credentials, json_body=register_payload, headers=headers)
        logger.info(f"Register repo response: {r.status_code} {r.text}")

        if r.status_code not in (200, 201):
            logger.error(f"Repo registration failed: {r.text}")
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Repo registration failed",
                    "details": r.text,
                    "status_code": r.status_code
                })
            }

        # ---------- CREATE SNAPSHOT ----------
        url_snap = f"https://{domain}/_snapshot/{repo_name}/{snapshot_name}"
        logger.info(f"Creating snapshot {snapshot_name} ...")

        r = signed_request("PUT", url_snap, region, service, credentials, headers=headers)
        logger.info(f"Create snapshot response: {r.status_code} {r.text}")

        if r.status_code not in (200, 201):
            logger.error(f"Snapshot creation failed: {r.text}")
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Snapshot creation failed",
                    "details": r.text,
                    "status_code": r.status_code
                })
            }

        # ---------- POLL STATUS ----------
        time.sleep(10)
        url_status = f"https://{domain}/_snapshot/{repo_name}/{snapshot_name}/_status"
        r = signed_request("GET", url_status, region, service, credentials)
        logger.info(f"Snapshot status: {r.status_code} {r.text}")

        status_data = r.json() if r.status_code == 200 else {}

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Snapshot created successfully",
                "snapshot_name": snapshot_name,
                "repo_name": repo_name,
                "status": status_data
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }
