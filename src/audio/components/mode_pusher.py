import os
import sys
from src.audio.cloud_storage.s3_operations import S3Sync
from src.audio.logger import logging
from src.audio.exception import CustomException
from src.audio.constants import *
from src.audio.entity.artifact_entity import *

class ModelPusher:
    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        self.model_evaluation_artifacts = model_evaluation_artifacts
    
    def initiate_model_pusher(self):
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                # Ensure s3_model_folder_path uses 'models/' prefix
                if not s3_model_folder_path.endswith("models/"):
                    s3_model_folder_path = s3_model_folder_path.rstrip('/') + "/models/"
                logging.info(f"Using S3 model path: {s3_model_folder_path}")
                
                s3_sync = S3Sync()
                # Verify the trained model path exists
                if not os.path.exists(trained_model_path):
                    raise FileNotFoundError(f"Trained model path {trained_model_path} does not exist")
                s3_sync.sync_folder_to_s3(folder=trained_model_path, aws_bucket_url=s3_model_folder_path)
                message = "Model Pusher pushed the current Trained model to Production server storage"
                response = {
                    "is_model_pushed": True,
                    "s3_model_path": f"{s3_model_folder_path.rstrip('/')}/{MODEL_NAME}",
                    "message": message
                }
                logging.info(f"Model push response: {response}")
            else:
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                if not s3_model_folder_path.endswith("models/"):
                    s3_model_folder_path = s3_model_folder_path.rstrip('/') + "/models/"
                message = "Current Trained Model is not accepted as model in Production has better loss"
                response = {
                    "is_model_pushed": False,
                    "s3_model_path": s3_model_folder_path,
                    "message": message
                }
                logging.info(f"Model push response: {response}")
            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            logging.info(f"Model pusher completed! Artifacts: {model_pusher_artifacts}")
            return model_pusher_artifacts
        except Exception as e:
            logging.error(f"Error in model pusher: {str(e)}")
            raise CustomException(e, sys)