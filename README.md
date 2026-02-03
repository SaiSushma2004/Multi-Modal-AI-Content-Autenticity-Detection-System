🧠 Multi-Modal AI Content Authenticity Detection System
Detect Real vs AI-Generated Image, Audio, and Text Content

📌 Overview

With the rapid evolution of Generative AI, distinguishing real content from AI-generated content across different modalities has become critical. Deepfakes, synthetic voices, and machine-generated text pose serious risks in misinformation, fraud, and digital trust.
This project implements an end-to-end Multi-Modal AI System that detects whether Image, Audio, or Text content is Real or AI-Generated using a combination of Deep Learning and Machine Learning models, deployed as a live cloud application.
The system provides a single unified Gradio-based interface, allowing users to upload content and instantly receive authenticity predictions.

🚀 Live Demo (Cloud Deployment)

🌐 Hugging Face Space (Live App):
👉 https://huggingface.co/spaces/sushma-ai/Multi-Modal-AI-Content-Authenticity-Detection-System

Users can directly upload image, audio, or text files and view real-time predictions without any local setup.

🎯 Key Features & Use Cases

🔍 Multi-Modal Detection
Image Authenticity Detection
Detects AI-generated vs real images using a CNN model
Supported formats: JPG, PNG, JPEG
Audio Authenticity Detection
Identifies real vs synthesized speech using audio feature extraction
Supported formats: WAV, MP3
Text Authenticity Detection
Classifies human-written vs AI-generated text using NLP techniques
Supported formats: TXT, PDF

⚡ User-Friendly Interface
Simple Gradio web UI
Upload → Select modality → Get prediction
Fast inference with confidence scores

🧩 Real-World Applications
Fake news and misinformation detection
Deepfake media verification
AI-generated content moderation
Academic integrity & plagiarism analysis
Digital forensics and research

🧠 Models Used
Modality	Model Type	Description
Image	CNN (TensorFlow/Keras)	Trained on real and AI-generated image datasets
Audio	ML Classifier (Scikit-learn)	Uses MFCC and spectral features extracted via Librosa
Text	NLP Model (TF-IDF + Classifier)	Detects AI-generated patterns in text

All trained models are saved and loaded during inference for efficient predictions.

🛠️ Tech Stack
Programming Language: Python 3.10+
Deep Learning: TensorFlow, Keras
Machine Learning: Scikit-learn
NLP: NLTK, TF-IDF
Audio Processing: Librosa, SoundFile
Image Processing: OpenCV, Pillow
UI Framework: Gradio
Cloud Platform: Hugging Face Spaces
Model Serialization: Joblib, Pickle
Data Handling: NumPy, Pandas

📂 Project Structure
├── app.py                   # Main Gradio application
├── train_image.py           # Image model training script
├── train_audio.py           # Audio model training script
├── train_text.py            # Text model training script
├── test_image.py            # Image model testing
├── test_audio.py            # Audio model testing
├── test_text.py             # Text model testing
├── dataset/                 # Training datasets
│   ├── images/
│   ├── audio/
│   └── text/
├── model/                   # Saved trained models & vectorizers
├── utils/                   # Preprocessing & helper functions
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

⚙️ Setup & Installation (Local)
1️⃣ Clone the Repository
git clone https://github.com/yourusername/ai-content-authenticity-detector.git
cd ai-content-authenticity-detector

2️⃣ Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ (Optional) Train Models from Scratch
python train_image.py
python train_audio.py
python train_text.py

Pre-trained models are already included for direct inference.

5️⃣ Run the Application
python app.py

The Gradio interface will launch locally in your browser.

☁️ Cloud Deployment (Hugging Face Spaces)
Why Hugging Face Spaces?

Free & fast AI deployment
Native support for Gradio
Easy sharing via public live links
No DevOps or server management needed

Deployment Steps

Create a new Hugging Face Space
Select Gradio as SDK
Upload:
app.py
requirements.txt
model/ directory

App auto-builds and goes live 🎉

🧪 Input Formats Supported
Modality	File Types
Image	JPG, PNG, JPEG
Audio	WAV, MP3
Text	TXT, PDF

🔮 Future Enhancements
Video deepfake detection
Large Language Model (LLM) based text detection
API endpoints for integration
Confidence explainability (XAI)
Multi-language text support

👩‍💻 Author
M.Sai Sushma 
B.Tech CSE (AI & ML)
AI | Machine Learning | Cloud Deployment
🔗 LinkedIn: https://www.linkedin.com/in/sai-sushma-maruboyina-382b34334?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
