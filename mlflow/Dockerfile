FROM python:3.10-slim

# MLflow + MinIO S3 deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

WORKDIR /mlflow_code
