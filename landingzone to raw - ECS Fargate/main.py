import json
import os
import sys
from video_processor import VideoProcessor


def main():
    """
    Entry point for ECS Fargate task
    Reads S3 bucket and key from environment variables set by Step Functions
    """
    try:
        # Get input from environment variables (passed by Step Functions)
        bucket = os.environ.get("BUCKET_NAME")
        key = os.environ.get("VIDEO_KEY")
        dst_bucket = os.environ.get("DESTINATION_BUCKET")

        if not bucket or not key or not dst_bucket:
            raise ValueError(
                "Missing required environment variables: BUCKET_NAME, VIDEO_KEY, and DESTINATION_BUCKET"
            )

        print(f"Starting video preprocessing...")
        print(f"Bucket: {bucket}")
        print(f"Key: {key}")
        print(f"Destination Bucket: {dst_bucket}")

        # Process video
        processor = VideoProcessor()
        result = processor.process_video(bucket, key, dst_bucket)

        # Write result to a file that Step Functions can read (optional)
        output_file = "/tmp/result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'=' * 60}")
        print("RESULT")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2))

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
