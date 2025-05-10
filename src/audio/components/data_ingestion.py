import os
import sys
import zipfile
from src.audio.constants import *
from src.audio.logger import logging
from src.audio.exception import CustomException
from src.audio.entity.config_entity import DataIngestionConfig
from src.audio.entity.artifact_entity import DataIngestionArtifacts
from src.audio.cloud_storage.s3_operations import S3Sync

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
            self.data_ingestion_artifact = self.data_ingestion_config.data_ingestion_artifact_dir
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_from_cloud(self) -> None:
        try:
            logging.info("Initiating data download from S3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            zip_file_path = self.data_ingestion_config.zip_data_path  # Path to speech_commands.zip

            # Check if the zip file already exists
            if os.path.exists(zip_file_path):
                logging.info(f"Data is already available at {zip_file_path}. Skipping download.")
            else:
                os.makedirs(download_dir, exist_ok=True)
                logging.info(f"Downloading data from {S3_BUCKET_DATA_URI} to {download_dir}")
                self.s3_sync.sync_folder_from_s3(
                    folder=download_dir,
                    aws_bucket_url=S3_BUCKET_DATA_URI
                )
                logging.info(f"Data downloaded from S3 to {zip_file_path}")

        except Exception as e:
            logging.error(f"Failed to download data from S3: {str(e)}")
            raise CustomException(e, sys)

    def unzip_data(self) -> None:
        try:
            logging.info("Unzipping the downloaded zip file...")
            raw_zip_path = self.data_ingestion_config.zip_data_path
            unzip_dir = self.data_ingestion_config.unzip_data_dir
            data_dir = os.path.join(unzip_dir, "data")  # Create path for data folder

            if os.path.isdir(data_dir):
                logging.info(f"Unzipped folder already exists at {data_dir}. Skipping unzipping.")
            else:
                os.makedirs(data_dir, exist_ok=True)
                with zipfile.ZipFile(raw_zip_path, "r") as f:
                    f.extractall(data_dir)
                logging.info(f"Unzipping completed. Extracted to {data_dir}")

        except Exception as e:
            logging.error(f"Failed to unzip data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Initiating the data ingestion component...")
            os.makedirs(self.data_ingestion_artifact, exist_ok=True)
            self.get_data_from_cloud()
            self.unzip_data()
            data_ingestion_artifact = DataIngestionArtifacts(
                data_folder_path=self.data_ingestion_config.unzip_data_dir
            )
            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Failed to complete data ingestion: {str(e)}")
            raise CustomException(e, sys)
