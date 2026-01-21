import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "model/image_model.h5"
IMAGE_SIZE = (128, 128)

model = load_model(MODEL_PATH)

img_path ="dataset/images/RealArt/-man-sits-with-a-woman-on-her-phone-at-a-table-while-looking-at-a-computer_l.jpg"   # Your test image

img = cv2.imread(img_path)
img = cv2.resize(img, IMAGE_SIZE)
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]

print("\n🔎 Prediction Score:", prediction)

if prediction >= 0.5:
    print("🟢 REAL Image")
else:
    print("🔴 FAKE (AI-generated) Image")
