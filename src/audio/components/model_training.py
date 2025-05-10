import os
import sys
import joblib
import torch
import torchaudio
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from src.audio.constants import *
from src.audio.logger import logging
from src.audio.exception import CustomException
from src.audio.entity.config_entity import *
from src.audio.entity.artifact_entity import *
from src.audio.entity.custom_model import *
from src.audio.utils import *

class ModelTraining:
    def __init__(self, data_transformation_artifact : DataTransformationArtifacts,
    model_trainer_config : ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def get_data_loader(self, train_data):
        try:
            val_size = int(len(train_data) * 0.2)
            train_size = len(train_data) - val_size

            logging.info("Shuffle and split the training and validation set")
            train_ds, val_ds = random_split(train_data, [train_size, val_size])

            # PyTorch data loaders
            training_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=True)
            valid_dl = DataLoader(val_ds, BATCH_SIZE*2, num_workers=0,pin_memory=True)

            logging.info("Exit get_data_loader method of model trainer")
            print(len(training_dl))
            print(len(valid_dl))

            return training_dl, valid_dl

        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self,train_data):
        try:
            logging.info("getting the pre-trained resnet model")

            num_classes = len(train_data.classes)

            model = ResNet9(3,num_classes)

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def load_to_GPU(self, training_dl, valid_dl, model):
        try:
            logging.info('loading model to GPU')
            DEVICE = get_default_device()

            model = to_device(model, DEVICE)

            logging.info('loading data to GPU')
            training_dl = DeviceDataLoader(training_dl, DEVICE)
            valid_dl = DeviceDataLoader(valid_dl, DEVICE)

            logging.info("loading data and model to GPU is done")
            return training_dl, valid_dl, model
        except Exception as e:
            raise CustomException(e,sys)

    def train_model(self, model, train_dl, valid_dl):
        try:
            logging.info("Model training started")
            fitted_model , result = my_fit_method(epochs=EPOCHS, lr=LEARNING_RATE, model=model, train_data_loader=train_dl, val_loader=valid_dl, opt_func=OPTIMIZER,
            grad_clip = GRAD_CLIP)
            logging.info("Model training done")
            return fitted_model , result 
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        try:
            train_transform = tt.Compose([
                tt.Resize((201,81)),
                tt.ToTensor()
            ])  
            os.makedirs(self.model_trainer_config.model_trainer_artifact_dir,exist_ok=True)
            logging.info("Saving transformer object for prediction")
            joblib.dump(train_transform, self.model_trainer_config.transformer_object_path)

            train_data = ImageFolder(self.data_transformation_artifact.images_folder_path,
            transform = train_transform)
            
            train_dl, valid_dl = self.get_data_loader(train_data)
            logging.info("load the model")

            logging.info("load the model")
            model = self.get_model(train_data)
            torch.cuda.empty_cache()

            logging.info("loading requirements to GPU")
            training_dl, valid_dl, model = self.load_to_GPU(train_dl, valid_dl, model)

            fitted_model, result  = self.train_model(model=model, train_dl=training_dl, valid_dl=valid_dl)
   
            logging.info(f"saving the model at {self.model_trainer_config.model_path}")
            torch.save(model.state_dict(), self.model_trainer_config.model_path)

            model_trainer_artifact = ModelTrainerArtifacts(
                model_path=self.model_trainer_config.model_path,
                result=result,
                transformer_object_path=self.model_trainer_config.transformer_object_path
            )
            logging.info(f"modler trainer artifact {model_trainer_artifact}")
            logging.info("model training completed")
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

        