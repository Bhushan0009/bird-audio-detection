import os
import keras
import librosa
import warnings
import tensorflow
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request


app = Flask(__name__)
 
 

final = pd.read_pickle("extracted_df.pkl")
y = np.array(final["name"].tolist())
le = LabelEncoder()
le.fit_transform(y)
Model1_ANN = load_model("Model1.h5")


def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])


def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')  
    
@app.route("/index", methods=['GET'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		audio_path = request.files['wavfile']

		img_path = "static/tests/" + audio_path.filename	
		audio_path.save(img_path)
	 
		predict_result =  ANN_print_prediction(img_path)

	return render_template("prediction.html", prediction = predict_result, audio_path= img_path)

 
@app.route("/chart")
def chart():
	return render_template('chart.html')     
 
if __name__ =='__main__':
	app.run(debug = True)
