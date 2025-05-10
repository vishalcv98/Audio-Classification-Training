import os,sys
import joblib
import torchaudio
from PIL import Image
import matplotlib.pyplot as plt
from src.audio.cloud_storage.s3_operations import S3Sync
from src.audio.exception import CustomException
from src.audio.constants import *
from src.audio.entity.config_entity import *
from src.audio.utils import *
from src.audio.entity.custom_model import *

DEVICE = get_default_device()

class SinglePrediction:
    def __init__(self):
        try:
            self.s3_sync = S3Sync()
            self.prediction_config = PredictionPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_model_in_production(self):
        try:
            s3_model_path = self.prediction_config.s3_model_path
            model_download_path = self.prediction_config.prediction_artifact_dir
            os.makedirs(model_download_path, exist_ok=True)
            if len(os.listdir(model_download_path)) == 0:
                self.s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        try:
            self.get_model_in_production()

            prediction_model_path = self.prediction_config.model_download_path

            prediction_model = to_device(ResNet9(3,NUM_CLASSES), DEVICE)

            prediction_model.load_state_dict(torch.load(prediction_model_path, map_location=torch.device('cpu')))

            # for gpu devices
            # prediction_model.load_state_dict(torch.load(prediction_model_path))

            prediction_model.eval()

            return prediction_model
        except Exception as e:
            raise CustomException(e, sys)

    def get_audio_waveform_sr(self,filename):
        waveform, _ = torchaudio.load(filename)
        return waveform


    def create_spectrogram_images(self):
        audio_path_dir = self.prediction_config.audio_path_dir
        for file in os.listdir(audio_path_dir):
            if file.endswith(".wav"):
                print(os.path.join(audio_path_dir, file)) 
                filename = os.path.join(audio_path_dir, file)
                
        waveform = self.get_audio_waveform_sr(filename=filename)
       
        image_save_path = os.path.join('static','upload','image.jpg')
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        plt.imsave(image_save_path, spectrogram.log2()[0,:,:].numpy(), cmap='viridis')

    def _get_image_tensor(self, image_path):
        try:
            img = Image.open(image_path)
            transforms = joblib.load(self.prediction_config.transforms_path)
            img_tensor = transforms(img)
            return img_tensor
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self):
        try:
            self.get_model_in_production()
            
            model = self.get_model()

            self.create_spectrogram_images()

            image = self._get_image_tensor(self.prediction_config.image_path)

            result = predict_image(image, model, DEVICE, NUM_CLASSES)

            return result

        except Exception as e:
            raise CustomException(e, sys)