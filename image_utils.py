import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMAGE_SIZE = (128, 128)
model = load_model("model/image_model.h5")

def predict_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    return "🟢 REAL Image" if pred >= 0.5 else "🔴 AI-GENERATED Image"
