import json
import boto3
import subprocess
import os
import math
from datetime import datetime


class VideoProcessor:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.MAX_SIZE_GB = 2.0
        self.MAX_DURATION_MINUTES = 120
        self.FFPROBE_PATH = "ffprobe"
        self.FFMPEG_PATH = "ffmpeg"

    def process_video(self, bucket, key, dst_bucket):
        """Main function to process and split video"""
        try:
            print(f"Processing video: s3://{bucket}/{key}")

            # Step 1: Check file size
            print("\nStep 1: Checking file size...")
            size_response = self.s3_client.head_object(Bucket=bucket, Key=key)
            size_bytes = size_response["ContentLength"]
            size_gb = size_bytes / (1024**3)
            size_mb = size_bytes / (1024**2)

            print(f"File size: {size_gb:.2f} GB ({size_mb:.2f} MB)")

            # Step 2: Download video
            print("\nStep 2: Downloading video...")
            local_file = "/tmp/video.mp4"
            self.s3_client.download_file(bucket, key, local_file)
            print(f"✓ Downloaded {size_mb:.2f} MB")

            # Step 3: Get metadata
            print("\nStep 3: Analyzing metadata...")
            metadata = self.get_video_metadata(local_file)

            duration_seconds = metadata["duration"]
            duration_minutes = duration_seconds / 60
            bitrate_mbps = metadata["bitrate_mbps"]

            print(
                f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)"
            )
            print(f"Bitrate: {bitrate_mbps:.2f} Mbps")

            # Step 4: Check if splitting needed
            size_exceeds = size_gb > self.MAX_SIZE_GB
            duration_exceeds = duration_minutes > self.MAX_DURATION_MINUTES

            if not size_exceeds and not duration_exceeds:
                print("\n✅ NO SPLITTING NEEDED")
                self.cleanup_tmp()

                return {
                    "splitting_needed": False,
                    "size_gb": round(size_gb, 2),
                    "duration_minutes": round(duration_minutes, 2),
                    "bucket": bucket,
                    "key": key,
                }

            # Step 5: Plan segments
            print("\n" + "=" * 60)
            print("PLANNING SEGMENTS")
            print("=" * 60)

            segments = self.plan_segments(
                total_duration=duration_seconds,
                total_size_mb=size_mb,
                bitrate_mbps=bitrate_mbps,
                max_duration_minutes=self.MAX_DURATION_MINUTES,
                max_size_gb=self.MAX_SIZE_GB,
            )

            print(f"\n✓ Planned {len(segments)} segments")

            # Step 6: Split and upload
            print("\n" + "=" * 60)
            print("SPLITTING AND UPLOADING")
            print("=" * 60)

            final_parts = self.split_and_upload_segments(
                local_file=local_file,
                segments=segments,
                dst_bucket=dst_bucket,
                original_key=key,
            )

            self.cleanup_tmp()

            print("\n✅ COMPLETE")

            return {
                "splitting_needed": True,
                "reason": "size" if size_exceeds else "duration",
                "original": {
                    "bucket": bucket,
                    "key": key,
                    "size_gb": round(size_gb, 2),
                    "duration_minutes": round(duration_minutes, 2),
                },
                "parts": final_parts,
                "total_parts": len(final_parts),
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.cleanup_tmp()
            raise

    def get_video_metadata(self, file_path):
        """Get video metadata"""
        cmd = [
            self.FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate,size",
            "-of",
            "json",
            file_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise Exception(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        duration = float(data["format"].get("duration", 0))
        bitrate = float(data["format"].get("bit_rate", 0))
        size_bytes = float(data["format"].get("size", 0))

        bitrate_mbps = (
            bitrate / 1_000_000
            if bitrate > 0
            else (size_bytes * 8) / (duration * 1_000_000)
        )

        return {
            "duration": duration,
            "bitrate": bitrate,
            "bitrate_mbps": bitrate_mbps,
            "size_bytes": size_bytes,
        }

    def plan_segments(
        self,
        total_duration,
        total_size_mb,
        bitrate_mbps,
        max_duration_minutes,
        max_size_gb,
    ):
        """Plan equal-duration segments"""
        max_duration_seconds = max_duration_minutes * 60
        max_size_mb = max_size_gb * 1024

        max_duration_by_size = (
            (max_size_mb * 8) / bitrate_mbps if bitrate_mbps > 0 else float("inf")
        )
        target_segment_duration = min(max_duration_seconds, max_duration_by_size) * 0.9

        num_segments = math.ceil(total_duration / target_segment_duration)
        actual_segment_duration = total_duration / num_segments

        print(
            f"Target: {target_segment_duration:.1f}s, Segments: {num_segments}, Actual: {actual_segment_duration:.1f}s"
        )

        segments = []
        for i in range(num_segments):
            start = i * actual_segment_duration
            end = (i + 1) * actual_segment_duration
            duration = actual_segment_duration

            estimated_size_mb = (duration * bitrate_mbps) / 8

            segments.append(
                {
                    "index": i + 1,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "estimated_size_mb": estimated_size_mb,
                }
            )

        return segments

    def split_and_upload_segments(self, local_file, segments, dst_bucket, original_key):
        """Split and upload segments"""
        base_name = os.path.splitext(original_key)[0]
        extension = os.path.splitext(original_key)[1]

        final_parts = []

        for segment in segments:
            idx = segment["index"]
            start = segment["start"]
            duration = segment["duration"]

            segment_file = f"/tmp/segment_{idx}.mp4"

            print(
                f"\nSegment {idx}/{len(segments)}: {start:.1f}s - {start + duration:.1f}s"
            )

            # Split
            cmd = [
                self.FFMPEG_PATH,
                "-ss",
                str(start),
                "-i",
                local_file,
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-fflags",
                "+genpts",
                "-movflags",
                "+faststart",
                "-y",
                segment_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"❌ Failed: {result.stderr}")
                continue

            actual_size_mb = os.path.getsize(segment_file) / (1024**2)
            print(f"✓ Created: {actual_size_mb:.2f} MB")

            # Upload
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m")
            day = now.strftime("%d")

            segment_key = (
                f"year={year}/month={month}/day={day}/{base_name}_part{idx}{extension}"
            )
            print(f"Uploading to s3://{dst_bucket}/{segment_key}...")
            self.s3_client.upload_file(segment_file, dst_bucket, segment_key)
            print(f"✓ Uploaded")

            os.remove(segment_file)

            final_parts.append(
                {
                    "part": idx,
                    "bucket": dst_bucket,
                    "key": segment_key,
                    "s3_url": f"s3://{dst_bucket}/{segment_key}",
                    "size_mb": round(actual_size_mb, 2),
                    "duration_seconds": round(duration, 2),
                    "duration_minutes": round(duration / 60, 2),
                    "start_time": round(start, 2),
                    "end_time": round(start + duration, 2),
                }
            )

        return final_parts

    def cleanup_tmp(self):
        """Cleanup temp files"""
        import glob

        patterns = ["/tmp/video.mp4", "/tmp/segment_*.mp4"]

        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Cleanup warning: {e}")
