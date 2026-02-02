import gradio as gr
import numpy as np
import os
import tempfile
import joblib
import librosa
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------ CONSTANTS ------------------
IMAGE_SIZE = (128, 128)
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40

# ------------------ PATHS ------------------
IMAGE_MODEL_PATH = "model/image_model.h5"
AUDIO_MODEL_PATH = "model/audio_model.h5"
TEXT_MODEL_PATH = "model/text_model.pkl"
TEXT_VECTORIZER_PATH = "model/text_vectorizer.pkl"

# ------------------ SAFE LOADERS ------------------
def load_image_model():
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(f"Missing {IMAGE_MODEL_PATH}")
    return load_model(IMAGE_MODEL_PATH)

def load_audio_model():
    if not os.path.exists(AUDIO_MODEL_PATH):
        raise FileNotFoundError(f"Missing {AUDIO_MODEL_PATH}")
    return load_model(AUDIO_MODEL_PATH)

def load_text_model():
    if not os.path.exists(TEXT_MODEL_PATH) or not os.path.exists(TEXT_VECTORIZER_PATH):
        raise FileNotFoundError("Missing text model/vectorizer")
    return joblib.load(TEXT_MODEL_PATH), joblib.load(TEXT_VECTORIZER_PATH)

# ------------------ AUDIO FEATURES ------------------
def extract_audio_features(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, N_MFCC, 1)

# ------------------ PREDICTIONS ------------------
def predict_image(img):
    model = load_image_model()
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]
    return f"{p:.4f} | {'REAL' if p >= 0.5 else 'AI-GENERATED'}"

def predict_audio(audio_path):
    model = load_audio_model()
    features = extract_audio_features(audio_path)
    p = model.predict(features)[0][0]
    return f"{p:.4f} | {'REAL' if p >= 0.5 else 'AI-GENERATED'}"

def predict_text(text):
    model, vectorizer = load_text_model()
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "REAL" if pred == 0 else "AI-GENERATED"

# ------------------ UI ------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🛡️ AI Content Authenticity Detector")

    with gr.Tab("Image"):
        img = gr.Image(type="pil")
        out = gr.Textbox()
        gr.Button("Predict").click(predict_image, img, out)

    with gr.Tab("Audio"):
        aud = gr.Audio(type="filepath",label="Upload Audio WAV/MP3")
        out2 = gr.Textbox(label="Prediction")
        gr.Button("Predict").click(predict_audio, aud, out2)

    with gr.Tab("Text"):
        txt = gr.Textbox(lines=6)
        out3 = gr.Textbox()
        gr.Button("Predict").click(predict_text, txt, out3)

demo.launch()
