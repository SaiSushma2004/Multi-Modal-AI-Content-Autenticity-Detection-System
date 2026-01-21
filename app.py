import streamlit as st
from utils.image_utils import predict_image
from utils.audio_utils import predict_audio
from utils.text_utils import predict_text

st.set_page_config(page_title="AI Authenticity Detector", layout="centered")

st.title("🔍 AI Content Authenticity Detector")
st.write("Detect whether **Image / Audio / Text** is REAL or AI-GENERATED")

# ------------------------------
# SELECT INPUT TYPE
# ------------------------------
option = st.selectbox(
    "What do you want to verify?",
    ("Select", "Image", "Audio", "Text")
)

# ------------------------------
# IMAGE
# ------------------------------
if option == "Image":
    st.subheader("🖼 Upload Image")
    image_file = st.file_uploader(
        "Upload JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Check Image"):
            result = predict_image(image_file)
            st.success(f"Result: {result}")

# ------------------------------
# AUDIO
# ------------------------------
elif option == "Audio":
    st.subheader("🎵 Upload Audio")
    audio_file = st.file_uploader(
        "Upload WAV or MP3 audio",
        type=["wav", "mp3"]
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("Check Audio"):
            result = predict_audio(audio_file)
            st.success(f"Result: {result}")

# ------------------------------
# TEXT
# ------------------------------
elif option == "Text":
    st.subheader("📄 Upload Text File")
    text_file = st.file_uploader(
        "Upload TXT file",
        type=["txt"]
    )

    if text_file:
        text_content = text_file.read().decode("utf-8")
        st.text_area("Text Content", text_content, height=200)

        if st.button("Check Text"):
            result = predict_text(text_content)
            st.success(f"Result: {result}")
