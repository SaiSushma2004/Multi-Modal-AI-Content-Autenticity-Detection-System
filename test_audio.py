import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "model/audio_model.h5"
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLES:
        padding = SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

model = load_model(MODEL_PATH)

file_path = "dataset/audio/Real/biden-original.wav"   # your test file
features = extract_features(file_path)
features = np.expand_dims(features, axis=0)
features = np.expand_dims(features, axis=-1)

prediction = model.predict(features)[0][0]

print("\n🔎 Prediction Score:", prediction)

if prediction >= 0.5:
    print("🟢 REAL Audio")
else:
    print("🔴 FAKE Audio")
