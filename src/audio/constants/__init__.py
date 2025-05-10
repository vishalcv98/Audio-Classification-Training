import os
import torch
from datetime import datetime

ARTIFACTS_DIR: str = "artifacts"
S3_BUCKET_DATA_URI = "s3://speech-classifierr/data.zip"
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion"
S3_DATA_FOLDER_NAME: str = "data.zip"
UNZIPPED_FOLDER_NAME: str = "unzip"
DATA_DIR_NAME: str = "data"

# constants related to data tranformation
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "data_transformation"
IMAGES_DIR:str = "spectrograms"
TEST_DIR: str = "test"
SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0

# constants related to model training
MODEL_TRAINER_ARTIFACTS_DIR : str = 'model_training'
MODEL_NAME: str = "model.pt"
TRANSFORM_OBJECT_NAME: str = "transform.pkl"
BATCH_SIZE: int = 15
EPOCHS: int = 15
LEARNING_RATE:float = 0.001
GRAD_CLIP : float = 0.1
WEIGHT_DECAY : float = 1e-4
IN_CHANNELS: int = 3
OPTIMIZER = torch.optim.RMSprop
NUM_CLASSES :int = 2

# constants realted to model evaluation 
S3_BUCKET_MODEL_URI: str = "s3://speech-classifierr/model/models"
MODEL_EVALUATION_DIR: str = "model_evaluation"
S3_MODEL_DIR_NAME: str = "s3_model"
S3_MODEL_NAME: str = "model.pt"
BASE_LOSS: int = 1.00

# constants realted to prediction pipeline
PREDICTION_PIPELINE_DIR_NAME: str = "prediction_artifact"
IMAGE_NAME : str = 'image.jpg'
STATIC_DIR ="static"
MODEL_SUB_DIR = 'model'
UPLOAD_SUB_DIR = 'upload'
