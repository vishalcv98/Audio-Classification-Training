import os, sys
from flask import Flask, render_template, request
from src.audio.exception import CustomException
from src.audio.pipeline.prediction_pipeline import SinglePrediction
from src.audio.pipeline.training_pipeline import TrainingPipeline
from src.audio.constants import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def land():
    return render_template("landing.html")

@app.route('/index', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            upload_file = request.files['fileup']
            filename = 'upload_file.wav'
            upload_audio_path = os.path.join(os.getcwd(), STATIC_DIR, UPLOAD_SUB_DIR)
            os.makedirs(upload_audio_path, exist_ok=True)
            upload_audio_path = os.path.join(os.getcwd(), STATIC_DIR, UPLOAD_SUB_DIR, filename)
            upload_file.save(upload_audio_path)
            pred = SinglePrediction()
            result = pred.predict()
            return render_template("index.html", upload=True, upload_audio=filename, text=result.upper())
        return render_template("index.html")
    
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/training', methods=['GET'])
def training():
    return render_template("training.html")

@app.route('/train', methods=['GET', 'POST'])
def train():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return "Training Completed"
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
