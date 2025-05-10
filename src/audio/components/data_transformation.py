import os
import sys
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.audio.constants import *
from src.audio.logger import logging
from src.audio.exception import CustomException
from src.audio.entity.config_entity import *
from src.audio.entity.artifact_entity import *

# Suppress PIL debug logs
logging.getLogger('PIL').setLevel(logging.WARNING)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifacts) -> None:
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            # Set device for GPU or CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
        except Exception as e:
            raise CustomException(e, sys)

    def load_audio_files(self, path: str, label: str):
        try:
            dataset = []
            walker = sorted(str(p) for p in Path(path).glob('*.wav'))
            if not walker:
                logging.warning(f"No .wav files found in {path}")
                return dataset
            for i, file_path in enumerate(walker):
                path, filename = os.path.split(file_path)
                speaker, _ = os.path.splitext(filename)
                try:
                    speaker_id, utterance_number = speaker.split("_nohash_")
                    utterance_number = int(utterance_number)
                except ValueError:
                    logging.warning(f"Invalid filename format: {filename}. Skipping.")
                    continue
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)
                dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])
            return dataset
        except Exception as e:
            raise CustomException(e, sys)

    def create_spectrogram_images(self, dataloader, label_dir, is_test=False):
        try:
            # Set directory based on whether it's test or train data
            base_dir = self.data_transformation_config.test_dir if is_test else self.data_transformation_config.images_dir
            directory = os.path.join(base_dir, label_dir)
            
            if os.path.isdir(directory):
                logging.info(f"Spectrogram directory exists for {label_dir}: {directory}")
            else:
                os.makedirs(directory, mode=0o777, exist_ok=True)
                logging.info(f"Created spectrogram directory: {directory}")
                
            spectrogram_count = 0
            for i, data in enumerate(dataloader):
                waveform = data[0].to(self.device)  # Move to GPU
                sample_rate = data[1][0]
                label = data[2]
                ID = data[3]

                # Create transformed waveforms
                spectrogram_transform = torchaudio.transforms.Spectrogram(
                    n_fft=400, win_length=None, hop_length=None, window_fn=torch.hann_window
                ).to(self.device)
                spectrogram_tensor = spectrogram_transform(waveform)
                
                # Convert to log scale and handle NaN/inf
                spectrogram_data = spectrogram_tensor[0].log2()
                spectrogram_data = torch.where(
                    torch.isfinite(spectrogram_data),
                    spectrogram_data,
                    torch.tensor(0.0, device=self.device)
                )
                path_to_save_img = os.path.join(directory, f"spec_img{i}.png")
                plt.imsave(path_to_save_img, spectrogram_data[0, :, :].cpu().numpy(), cmap='viridis')
               
                spectrogram_count += 1
            logging.info(f"Saved spectrogram: {path_to_save_img}")
            
            if spectrogram_count == 0:
                logging.warning(f"No spectrogram images generated for {label_dir} in {base_dir}")
            else:
                logging.info(f"Generated {spectrogram_count} spectrogram images for {label_dir} in {base_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Initiating the data transformation component...")
            # Define folder paths for dog and cat
            dog_folder_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'data', 'dog', '00f0204f_nohash_0.wav')
            cat_folder_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'data', 'cat', '00b01445_nohash_0.wav')

            # Verify folder existence
            logging.info(f"Dog folder exists: {os.path.exists(dog_folder_path)} ({dog_folder_path})")
            logging.info(f"Cat folder exists: {os.path.exists(cat_folder_path)} ({cat_folder_path})")

            # Log directory paths for debugging
            logging.info(f"Training images directory: {self.data_transformation_config.images_dir}")
            logging.info(f"Test images directory: {self.data_transformation_config.test_dir}")

            # Load datasets
            dog_dataset = self.load_audio_files(dog_folder_path[:-22], 'dog')
            cat_dataset = self.load_audio_files(cat_folder_path[:-22], 'cat')
            logging.info(f'Length of dog dataset: {len(dog_dataset)}')
            logging.info(f'Length of cat dataset: {len(cat_dataset)}')

            if len(dog_dataset) == 0 or len(cat_dataset) == 0:
                raise CustomException("No audio files found in dog or cat folders. Check the unzipped data structure.", sys)

            # Define test split ratio (20% for testing)
            test_split_ratio = 0.3
            num_test_dog = max(1, int(len(dog_dataset) * test_split_ratio))
            num_test_cat = max(1, int(len(cat_dataset) * test_split_ratio))
            logging.info(num_test_dog,num_test_cat)
            #num_test_cat=num_test_cat[0]

            # Split into train and test datasets
            train_dog = dog_dataset[:-num_test_dog]
            test_dog = dog_dataset[-num_test_dog:]
            train_cat = cat_dataset[:-num_test_cat]
            test_cat = cat_dataset[-num_test_cat:]

            logging.info(f'Length of dog training dataset: {len(train_dog)}')
            logging.info(f'Length of dog test dataset: {len(test_dog)}')
            logging.info(f'Length of cat training dataset: {len(train_cat)}')
            logging.info(f'Length of cat test dataset: {len(test_cat)}')

            if len(test_dog) == 0 or len(test_cat) == 0:
                raise CustomException("Test datasets are empty. Increase dataset size or adjust test_split_ratio.", sys)

            # Create DataLoaders with dynamic pin_memory
            trainloader_dog = torch.utils.data.DataLoader(
                train_dog, batch_size=1, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
            trainloader_cat = torch.utils.data.DataLoader(
                train_cat, batch_size=1, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
            testloader_dog = torch.utils.data.DataLoader(
                test_dog, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
            testloader_cat = torch.utils.data.DataLoader(
                test_cat, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )

            # Generate spectrogram images for train data
            self.create_spectrogram_images(trainloader_dog, 'dog', is_test=False)
            self.create_spectrogram_images(trainloader_cat, 'cat', is_test=False)

            # Generate spectrogram images for test data
            self.create_spectrogram_images(testloader_dog, 'dog', is_test=True)
            self.create_spectrogram_images(testloader_cat, 'cat', is_test=True)

            data_transformation_artifact = DataTransformationArtifacts(
                images_folder_path=self.data_transformation_config.images_dir,
                test_folder_path=self.data_transformation_config.test_dir
            )
            logging.info('Data transformation completed successfully.')
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)