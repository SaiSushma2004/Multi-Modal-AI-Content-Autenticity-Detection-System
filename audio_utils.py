import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/audio_model.h5")

def predict_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=(0, 2))

    pred = model.predict(mfcc)[0][0]
    return "🟢 REAL Audio" if pred >= 0.5 else "🔴 AI-GENERATED Audio"
