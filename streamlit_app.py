import streamlit as st
import numpy as np
import tempfile
import os
import joblib
import librosa
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    layout="centered"
)

st.title("🛡️ AI Content Authenticity Detector")
st.write("Detect whether **Image / Audio / Text** is **REAL or AI-GENERATED**")

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

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, N_MFCC, 1)

# ------------------ LOAD MODELS SAFELY ------------------
@st.cache_resource
def load_image_model():
    if not os.path.exists(IMAGE_MODEL_PATH):
        st.error("❌ Image model not found")
        st.stop()
    return load_model(IMAGE_MODEL_PATH)

@st.cache_resource
def load_audio_model():
    if not os.path.exists(AUDIO_MODEL_PATH):
        st.error("❌ Audio model not found")
        st.stop()
    return load_model(AUDIO_MODEL_PATH)

@st.cache_resource
def load_text_model():
    if not os.path.exists(TEXT_MODEL_PATH) or not os.path.exists(TEXT_VECTORIZER_PATH):
        st.error("❌ Text model/vectorizer not found")
        st.stop()
    model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    return model, vectorizer

# ------------------ MODE SELECTION ------------------
mode = st.selectbox(
    "Select content type",
    ["Image", "Audio", "Text"]
)

# =================================================
# IMAGE
# =================================================
if mode == "Image":
    st.subheader("🖼️ Image Authenticity Check")

    image_file = st.file_uploader(
        "Upload Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image = image.resize(IMAGE_SIZE)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        model = load_image_model()
        prediction = model.predict(image)[0][0]

        st.write("🔎 Prediction Score:", round(float(prediction), 4))

        if prediction >= 0.5:
            st.success("🟢 REAL Image")
        else:
            st.error("🔴 AI-GENERATED Image")

# =================================================
# AUDIO
# =================================================
elif mode == "Audio":
    st.subheader("🎧 Audio Authenticity Check")

    audio_file = st.file_uploader(
        "Upload Audio (MP3 / WAV)",
        type=["mp3", "wav"]
    )

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        try:
            model = load_audio_model()
            features = extract_audio_features(audio_path)
            prediction = model.predict(features)[0][0]

            st.write("🔎 Prediction Score:", round(float(prediction), 4))

            if prediction >= 0.5:
                st.success("🟢 REAL Audio")
            else:
                st.error("🔴 AI-GENERATED Audio")

        except Exception as e:
            st.error(f"❌ Audio processing failed: {e}")

        finally:
            os.remove(audio_path)

# =================================================
# TEXT
# =================================================
elif mode == "Text":
    st.subheader("📝 Text Authenticity Check")

    text_file = st.file_uploader("Upload Text File (.txt)", type=["txt"])
    text_input = st.text_area("Or paste text here")

    text = ""
    if text_file:
        text = text_file.read().decode("utf-8")
    elif text_input.strip():
        text = text_input

    if text.strip():
        model, vectorizer = load_text_model()
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("🔴 AI-GENERATED Text")
        else:
            st.success("🟢 REAL (Human-Written) Text")
