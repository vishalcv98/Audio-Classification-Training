import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import logging
from src.audio.logger import logging as logger

# Load environment variables from .env file
load_dotenv()

class S3Sync:
    def __init__(self):
        """Initialize S3 client using credentials from environment variables."""
        try:
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            logger.info(f"AWS_ACCESS_KEY_ID: {access_key[:4]}**** (masked for security)")
            logger.info(f"AWS_DEFAULT_REGION: {region}")

            if not access_key or 'EXAMPLE' in access_key:
                raise Exception("Invalid or placeholder AWS_ACCESS_KEY_ID in .env file")

            self.s3_client = boto3.client('s3', region_name=region)
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise Exception(f"Failed to initialize S3 client: {str(e)}")

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Upload all files in the specified folder to S3.

        :param folder: Local directory containing files to upload
        :param aws_bucket_url: S3 URL (e.g., s3://bucket_name/prefix/)
        """
        try:
            # Parse the aws_bucket_url (e.g., s3://speech-classifierr/models/)
            if not aws_bucket_url.startswith("s3://"):
                raise ValueError(f"Invalid S3 URL format: {aws_bucket_url}")
            
            # Remove 's3://' and split into bucket and prefix
            bucket_path = aws_bucket_url[5:].rstrip('/')
            bucket_name, *prefix_parts = bucket_path.split('/', 1)
            prefix = prefix_parts[0] if prefix_parts else ""
            
            logger.info(f"Uploading folder {folder} to s3://{bucket_name}/{prefix}")

            # Ensure the folder exists
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Folder {folder} does not exist")

            # Walk through the folder and upload each file
            uploaded_files = 0
            for root, _, files in os.walk(folder):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    # Calculate the relative path for the S3 key
                    relative_path = os.path.relpath(local_file_path, folder)
                    s3_key = os.path.join(prefix, relative_path).replace('\\', '/')
                    
                    # Check if file already exists in S3
                    try:
                        self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        logger.info(f"Overwriting existing file s3://{bucket_name}/{s3_key}")
                    except ClientError as e:
                        if e.response['Error']['Code'] != '404':
                            raise
                        logger.info(f"Uploading new file s3://{bucket_name}/{s3_key}")
                    
                    self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
                    uploaded_files += 1
            
            if uploaded_files == 0:
                logger.warning(f"No files found in {folder} to upload")
            else:
                logger.info(f"Successfully uploaded {uploaded_files} files to s3://{bucket_name}/{prefix}")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error uploading to S3: {error_code} - {error_message}")
            raise Exception(f"Error uploading to S3: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {str(e)}")
            raise Exception(f"Unexpected error uploading to S3: {str(e)}")

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Download a specific file or all files from an S3 folder to the local directory.
        
        :param folder: Local directory to save the file(s)
        :param aws_bucket_url: S3 URL (e.g., s3://speech-classifierr/data.zip or s3://speech-classifierr/model/)
        """
        try:
            # Parse the aws_bucket_url
            if not aws_bucket_url.startswith("s3://"):
                raise ValueError(f"Invalid S3 URL format: {aws_bucket_url}")
            
            # Remove 's3://' and split into bucket and key/prefix
            bucket_path = aws_bucket_url[5:].rstrip('/')
            bucket_name, *key_parts = bucket_path.split('/', 1)
            key = key_parts[0] if key_parts else ""
            
            # Ensure the local folder exists
            os.makedirs(folder, exist_ok=True)

            # Check if the URL points to a specific file (e.g., ends with .zip)
            if key and os.path.splitext(key)[1]:  # Has an extension, likely a file
                local_file_path = os.path.join(folder, os.path.basename(key))
                logger.info(f"Downloading single file s3://{bucket_name}/{key} to {local_file_path}")
                
                # Download the file
                self.s3_client.download_file(Bucket=bucket_name, Key=key, Filename=local_file_path)
                logger.info(f"Successfully downloaded s3://{bucket_name}/{key} to {local_file_path}")
            else:
                # Treat as a folder/prefix
                prefix = key + '/' if key else ""
                logger.info(f"Downloading from s3://{bucket_name}/{prefix} to {folder}")

                # List objects in the S3 bucket with the given prefix
                response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                if 'Contents' not in response:
                    logger.warning(f"No files found in s3://{bucket_name}/{prefix}")
                    return

                downloaded_files = 0
                for obj in response.get('Contents', []):
                    s3_key = obj['Key']
                    # Skip if it's a folder (ends with '/')
                    if s3_key.endswith('/'):
                        continue
                    # Calculate the local file path
                    relative_path = os.path.relpath(s3_key, prefix) if prefix else os.path.basename(s3_key)
                    local_file_path = os.path.join(folder, relative_path.replace('/', os.sep))
                    
                    # Ensure the local subdirectory exists
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    logger.info(f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}")
                    self.s3_client.download_file(Bucket=bucket_name, Key=s3_key, Filename=local_file_path)
                    downloaded_files += 1
                
                if downloaded_files == 0:
                    logger.warning(f"No files downloaded from s3://{bucket_name}/{prefix}")
                else:
                    logger.info(f"Successfully downloaded {downloaded_files} files from s3://{bucket_name}/{prefix} to {folder}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error downloading from S3: {error_code} - {error_message}")
            raise Exception(f"Error downloading from S3: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error downloading from S3: {str(e)}")
            raise Exception(f"Unexpected error downloading from S3: {str(e)}")
    
    def access_s3_file(self, bucket_name, file_key, region_name='us-east-1'):
        """
        Access a file from an S3 bucket.
        
        :param bucket_name: Name of the S3 bucket
        :param file_key: Key (path) of the file in the bucket
        :param region_name: AWS region of the bucket
        :return: File content as bytes or None if error occurs
        """
        try:
            # Get file from S3
            response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response['Body'].read()
            return file_content

        except ClientError as e:
            logger.error(f"Error accessing file from S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None