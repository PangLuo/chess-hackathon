"""
S3 Multipart Upload Script

This script provides functionality to upload large files and entire directories
to an Amazon S3 bucket using multipart uploads. It leverages the `boto3` library
to efficiently handle large files by splitting them into smaller chunks.

Functions:
- upload_file(file_path, key_name): Uploads a single file to S3 using multipart upload.
- upload_folder(folder_path, s3_prefix): Recursively uploads all files in a folder to S3.
- main(path, s3_prefix): Determines whether the given path is a file or folder and uploads accordingly.

Usage:
- Modify `input_path` and `s3_key_prefix` in the `__main__` block to specify the file or
  directory to be uploaded and its corresponding S3 key prefix.
- Run the script to initiate the upload process.

Dependencies:
- boto3 (AWS SDK for Python)

Example:
    python upload.py
"""

import os
import time

import boto3

s3_client = boto3.client("s3", region_name="us-east-1")
bucket_name = "gnn"


def upload_file(file_path, key_name):
    """Uploads a file to S3 using multipart upload."""
    multipart_upload = s3_client.create_multipart_upload(
        Bucket=bucket_name, Key=key_name
    )
    upload_id = multipart_upload["UploadId"]
    print(f"Started multipart upload for {file_path} with UploadId: {upload_id}")

    parts = []
    chunk_size = 100 * 1024 * 1024

    with open(file_path, "rb") as file:
        for part_number in range(1, 10000):
            start = time.time()
            data = file.read(chunk_size)
            if not data:
                break

            try:
                response = s3_client.upload_part(
                    Bucket=bucket_name,
                    Key=key_name,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data,
                )
            except Exception as e:
                print(f"Error uploading part {part_number} of {file_path}: {e}")
                s3_client.abort_multipart_upload(
                    Bucket=bucket_name, Key=key_name, UploadId=upload_id
                )
                return

            parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
            print(
                f"Uploaded part {part_number} of {file_path}. Time spent: {time.time() - start:.2f} seconds"
            )

    s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=key_name,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    print(f"Multipart upload completed successfully for {file_path}.")


def upload_folder(folder_path, s3_prefix):
    """Uploads all files in a folder to S3."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)
            key_name = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            upload_file(file_path, key_name)


def main(path, s3_prefix):
    if os.path.isfile(path):
        upload_file(path, s3_prefix)
    elif os.path.isdir(path):
        upload_folder(path, s3_prefix)
    else:
        print("Invalid path. Please provide a valid file or folder.")


if __name__ == "__main__":
    input_path = "/root/chess-hackathon/models/chessGNN/chess_dataset_hdf5"
    s3_key_prefix = "chess_dataset_hdf5"
    main(input_path, s3_key_prefix)
