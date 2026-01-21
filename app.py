import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import joblib
import librosa
from tensorflow.keras.models import load_model


SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40

def extract_audio_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)

    # Model expects shape (1, 40, 1)
    return mfcc.reshape(1, N_MFCC, 1)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Content Authenticity Detector",
    layout="centered"
)

st.title("🛡️ AI Content Authenticity Detector")
st.write("Detect whether **Image / Audio / Text** is **REAL or AI-GENERATED**")



# ------------------ PATHS ------------------
IMAGE_MODEL_PATH = "model/image_model.h5"
AUDIO_MODEL_PATH = "model/audio_model.h5"
TEXT_MODEL_PATH = "model/text_model.pkl"
TEXT_VECTORIZER_PATH = "model/text_vectorizer.pkl"

IMAGE_SIZE = (128, 128)

# ------------------ UTILS ------------------
@st.cache_resource
def load_image_model():
    return load_model(IMAGE_MODEL_PATH)

@st.cache_resource
def load_audio_model():
    return load_model(AUDIO_MODEL_PATH)

@st.cache_resource
def load_text_model():
    model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    return model, vectorizer

# ------------------ SELECT MODE ------------------
mode = st.selectbox(
    "Select content type",
    ["Image", "Audio", "Text"]
)

# =================================================
# IMAGE PREDICTION
# =================================================
if mode == "Image":
    st.subheader("🖼️ Image Authenticity Check")

    image_file = st.file_uploader(
        "Upload Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if image_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_file.read())
            img_path = tmp.name

        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        model = load_image_model()
        prediction = model.predict(img)[0][0]

        os.remove(img_path)

        st.write("🔎 Prediction Score:", float(prediction))

        if prediction >= 0.5:
            st.success("🟢 REAL Image")
        else:
            st.error("🔴 AI-GENERATED Image")

# =================================================
# AUDIO PREDICTION
# =================================================
elif mode == "Audio":
    st.subheader("🎧 Audio Authenticity Check")

    audio_file = st.file_uploader(
        "Upload Audio (MP3 / WAV)",
        type=["mp3", "wav"]
    )

    if audio_file is not None:
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
# TEXT PREDICTION
# =================================================
elif mode == "Text":
    st.subheader("📝 Text Authenticity Check")

    text_file = st.file_uploader(
        "Upload Text File (.txt)",
        type=["txt"]
    )

    text_input = st.text_area("Or paste text here")

    if text_file is not None:
        text = text_file.read().decode("utf-8")
    else:
        text = text_input

    if text.strip():
        model, vectorizer = load_text_model()
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("🔴 AI-GENERATED Text")
        else:
            st.success("🟢 REAL (Human-Written) Text")
