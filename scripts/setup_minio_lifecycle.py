#!/usr/bin/env python3
"""Configure MinIO bucket lifecycle policies for MLflow artifact retention.

Sets an S3-compatible lifecycle rule that automatically expires objects in the
MLflow artifacts bucket (``mlflow``) after a configurable number of days.
Objects expire at midnight UTC on the expiry date; MinIO removes them during
the next scheduled maintenance pass.

Usage (from the project root, with MinIO running):
    # Uses values from the environment (or .env via docker compose env):
    python scripts/setup_minio_lifecycle.py

    # Override the retention window:
    ARTIFACT_RETENTION_DAYS=30 python scripts/setup_minio_lifecycle.py

    # Point at a remote MinIO instance:
    MLFLOW_S3_ENDPOINT_URL=http://192.168.1.100:9000 \\
    MINIO_ROOT_USER=myadmin MINIO_ROOT_PASSWORD=mypassword \\
    python scripts/setup_minio_lifecycle.py

Environment variables:
    MLFLOW_S3_ENDPOINT_URL   — MinIO endpoint (default: http://localhost:9000)
    MINIO_ROOT_USER          — MinIO access key (default: minioadmin)
    MINIO_ROOT_PASSWORD      — MinIO secret key (default: minioadmin)
    ARTIFACT_RETENTION_DAYS  — Days before objects expire (default: 90)

Exit codes:
    0 — policy applied successfully
    1 — connection or API error
"""
import os
import sys

# boto3 is a transitive dependency of mlflow — always available in service containers.
try:
    import boto3
    from botocore.exceptions import ClientError, EndpointResolutionError
except ImportError:
    sys.exit("boto3 is required. Install with: pip install boto3")

# ---------------------------------------------------------------------------
# Configuration (read from environment with sensible local-dev defaults)
# ---------------------------------------------------------------------------
ENDPOINT_URL: str = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
ACCESS_KEY: str = os.environ.get("MINIO_ROOT_USER", "minioadmin")
SECRET_KEY: str = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET: str = "mlflow"
EXPIRE_DAYS: int = int(os.environ.get("ARTIFACT_RETENTION_DAYS", "90"))

# Rule ID stored in the lifecycle configuration (visible in MinIO console)
RULE_ID: str = "mlflow-artifact-retention"


def _create_s3_client() -> "boto3.client":
    """Create an S3 client pointed at the MinIO endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        # Disable virtual-hosted-style addressing — MinIO uses path-style
        config=boto3.session.Config(s3={"addressing_style": "path"}),
    )


def _ensure_bucket_exists(s3_client) -> None:
    """Create the bucket if it does not already exist."""
    try:
        s3_client.create_bucket(Bucket=BUCKET)
        print(f"[lifecycle] Created bucket: {BUCKET}")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"[lifecycle] Bucket already exists: {BUCKET}")
        else:
            raise


def _apply_lifecycle_policy(s3_client) -> None:
    """Write the expiration lifecycle rule to the bucket."""
    lifecycle_config = {
        "Rules": [
            {
                "ID": RULE_ID,
                "Status": "Enabled",
                # Empty prefix → rule applies to all objects in the bucket
                "Filter": {"Prefix": ""},
                "Expiration": {"Days": EXPIRE_DAYS},
            }
        ]
    }
    s3_client.put_bucket_lifecycle_configuration(
        Bucket=BUCKET,
        LifecycleConfiguration=lifecycle_config,
    )


def _verify_policy(s3_client) -> dict:
    """Read back the lifecycle configuration to confirm it was saved."""
    response = s3_client.get_bucket_lifecycle_configuration(Bucket=BUCKET)
    return response["Rules"][0]


def main() -> None:
    """Entry point — apply and verify the MinIO lifecycle policy."""
    print(f"[lifecycle] Endpoint : {ENDPOINT_URL}")
    print(f"[lifecycle] Bucket   : {BUCKET}")
    print(f"[lifecycle] Expires  : {EXPIRE_DAYS} days")

    try:
        s3 = _create_s3_client()
        _ensure_bucket_exists(s3)
        _apply_lifecycle_policy(s3)
        rule = _verify_policy(s3)
    except (ClientError, EndpointResolutionError, Exception) as exc:
        print(f"[lifecycle] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Confirm the rule that MinIO has stored
    print(
        f"[lifecycle] Applied rule '{rule['ID']}': "
        f"expire after {rule['Expiration']['Days']} days, "
        f"status={rule['Status']}"
    )
    print("[lifecycle] Done.")


if __name__ == "__main__":
    main()
