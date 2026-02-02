import gradio as gr
import numpy as np
import tempfile
import os
import joblib
import librosa
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------ CONSTANTS ------------------
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40
IMAGE_SIZE = (128, 128)

# ------------------ MODEL PATHS ------------------
IMAGE_MODEL_PATH = "model/image_model.h5"
AUDIO_MODEL_PATH = "model/audio_model.h5"
TEXT_MODEL_PATH = "model/text_model.pkl"
TEXT_VECTORIZER_PATH = "model/text_vectorizer.pkl"

# ------------------ LOAD MODELS ------------------
image_model = load_model(IMAGE_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)
text_model = joblib.load(TEXT_MODEL_PATH)
text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)

# ------------------ AUDIO FEATURE EXTRACTION ------------------
def extract_audio_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True
    )

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, N_MFCC, 1)

# ------------------ IMAGE PREDICTION ------------------
def predict_image(image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = image_model.predict(image)[0][0]
    label = "🟢 REAL Image" if prediction >= 0.5 else "🔴 AI-GENERATED Image"

    return f"Prediction Score: {prediction:.4f}\n{label}"

# ------------------ AUDIO PREDICTION ------------------
def predict_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file)
        path = tmp.name

    features = extract_audio_features(path)
    prediction = audio_model.predict(features)[0][0]
    os.remove(path)

    label = "🟢 REAL Audio" if prediction >= 0.5 else "🔴 AI-GENERATED Audio"
    return f"Prediction Score: {prediction:.4f}\n{label}"

# ------------------ TEXT PREDICTION ------------------
def predict_text(text):
    X = text_vectorizer.transform([text])
    pred = text_model.predict(X)[0]

    return "🟢 REAL (Human-Written) Text" if pred == 0 else "🔴 AI-GENERATED Text"

# ------------------ GRADIO UI ------------------
with gr.Blocks(title="AI Content Authenticity Detector") as demo:
    gr.Markdown("## 🛡️ AI Content Authenticity Detector")
    gr.Markdown("Detect whether **Image / Audio / Text** is **REAL or AI-GENERATED**")

    with gr.Tabs():
        with gr.Tab("🖼️ Image"):
            img_input = gr.Image(type="pil")
            img_output = gr.Textbox()
            gr.Button("Analyze Image").click(predict_image, img_input, img_output)

        with gr.Tab("🎧 Audio"):
            audio_input = gr.Audio(type="binary")
            audio_output = gr.Textbox()
            gr.Button("Analyze Audio").click(predict_audio, audio_input, audio_output)

        with gr.Tab("📝 Text"):
            text_input = gr.Textbox(lines=6, placeholder="Paste text here...")
            text_output = gr.Textbox()
            gr.Button("Analyze Text").click(predict_text, text_input, text_output)

demo.launch()

