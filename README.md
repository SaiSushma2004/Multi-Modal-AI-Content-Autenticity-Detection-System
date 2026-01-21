# Multi-Modal AI Content Authenticity Detection System

## Introduction
With the rapid advancement of generative AI, distinguishing real content from AI-generated content across multiple formats is becoming crucial. This project implements a **multi-modal AI system** capable of detecting whether images, audio, or text are real or AI-generated using a combination of deep learning and machine learning techniques.

The system provides a unified and user-friendly interface allowing users to upload files in supported formats and instantly get authenticity predictions, helping combat misinformation, deepfakes, and synthetic content proliferation.

---

## Features / Use Case
- Detect authenticity of **images** (formats: JPG, PNG) using a CNN model trained on real and AI-generated images.
- Analyze **audio files** (formats: WAV, MP3) by extracting audio features and classifying real vs synthesized speech.
- Classify **text content** (formats: TXT, PDF) using NLP techniques to identify human-written versus AI-generated text.
- User selects the input type and uploads files through a simple Streamlit web app.
- Real-time prediction with confidence scores to aid content verification and digital forensics.

---

## Technologies Used
- Python 3.10+
- TensorFlow / Keras (for image model)
- Scikit-learn (for audio and text models)
- Pandas & NumPy (data handling)
- OpenCV (image processing)
- Librosa (audio feature extraction)
- Streamlit (web UI)
- Joblib (model serialization)

---

## Setup and Installation

### 1. Clone the repository

git clone https://github.com/yourusername/ai-content-authenticity-detector.git
cd ai-content-authenticity-detector

2. Create and activate a virtual environment

python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

3. Install required dependencies

tensorflow
opencv-python
numpy
pandas
scikit-learn
streamlit
pillow
librosa
soundfile
matplotlib
nltk
pickle-mixin
joblib

4. Train models (optional)
If you want to retrain models from scratch:
python train_image.py
python train_audio.py
python train_text.py

5. Run the Streamlit application
streamlit run app.py
Project Structure

|-- .streamlit/
|    |-- config.toml
├── app.py                  # Main Streamlit app for user interaction
├── train_image.py          # Script to train image classification model
├── train_audio.py          # Script to train audio classification model
├── train_text.py           # Script to train text classification model
├── test_image.py           # Script to test image model separately
├── test_audio.py           # Script to test audio model separately
├── test_text.py            # Script to test text model separately
├── dataset/                # Datasets for training and testing
│   ├── images/
│   ├── audio/
│   └── text/
├── model/                  # Trained models and vectorizers
├── utils/                  # Helper utilities for preprocessing and prediction
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

6.Description and Use Case
This project aims to provide a reliable tool for verifying digital content authenticity, which is increasingly important in today's digital ecosystem to:
Combat misinformation and fake news.
Prevent misuse of AI-generated media.
Assist researchers and analysts in detecting synthetic content.
Help platforms moderate uploaded content efficiently.

The modular design allows easy extension to new modalities and integration with production systems.
