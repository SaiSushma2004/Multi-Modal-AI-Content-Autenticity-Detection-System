import gradio as gr
import numpy as np
import os
import joblib
import librosa
from PIL import Image
from tensorflow.keras.models import load_model

# =========================================================
# BASE PATHS (VERY IMPORTANT FOR HUGGING FACE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "image_model.h5")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_model.h5")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer.pkl")

# =========================================================
# CONSTANTS
# =========================================================
IMAGE_SIZE = (128, 128)
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40

# =========================================================
# AUDIO FEATURE EXTRACTION
# =========================================================
def extract_audio_features(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, N_MFCC, 1)

# =========================================================
# PREDICTION FUNCTIONS (SAFE + ERROR HANDLING)
# =========================================================
def predict_image(img):
    try:
        if img is None:
            return "❌ Please upload an image"

        if not os.path.exists(IMAGE_MODEL_PATH):
            return "❌ Image model not found"

        model = load_model(IMAGE_MODEL_PATH)

        img = img.resize(IMAGE_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        p = float(model.predict(img)[0][0])

        label = "🟢 REAL Image" if p >= 0.5 else "🔴 AI-GENERATED Image"
        return f"Prediction Score: {p:.4f}\n{label}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_audio(audio_path):
    try:
        if audio_path is None:
            return "❌ Please upload an audio file"

        if not os.path.exists(AUDIO_MODEL_PATH):
            return "❌ Audio model not found"

        model = load_model(AUDIO_MODEL_PATH)

        features = extract_audio_features(audio_path)
        p = float(model.predict(features)[0][0])

        label = "🟢 REAL Audio" if p >= 0.5 else "🔴 AI-GENERATED Audio"
        return f"Prediction Score: {p:.4f}\n{label}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_text(text):
    try:
        if text is None or text.strip() == "":
            return "❌ Please enter some text"

        if not os.path.exists(TEXT_MODEL_PATH) or not os.path.exists(TEXT_VECTORIZER_PATH):
            return "❌ Text model or vectorizer not found"

        model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(TEXT_VECTORIZER_PATH)

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        return "🟢 REAL (Human-written Text)" if pred == 0 else "🔴 AI-GENERATED Text"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("## 🛡️ AI Content Authenticity Detector")
    gr.Markdown("Detect whether **Image / Audio / Text** is **REAL or AI-GENERATED**")

    with gr.Tab("🖼️ Image"):
        img = gr.Image(type="pil", label="Upload Image")
        out_img = gr.Textbox(label="Prediction")
        gr.Button("Predict").click(predict_image, img, out_img)

    with gr.Tab("🎧 Audio"):
        aud = gr.Audio(type="filepath", label="Upload Audio (WAV / MP3)")
        out_aud = gr.Textbox(label="Prediction")
        gr.Button("Predict").click(predict_audio, aud, out_aud)

    with gr.Tab("📝 Text"):
        txt = gr.Textbox(lines=6, label="Enter Text")
        out_txt = gr.Textbox(label="Prediction")
        gr.Button("Predict").click(predict_text, txt, out_txt)

demo.launch()
