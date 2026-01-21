import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping

# =======================
# CONFIGURATION
# =======================
AUDIO_PATH = "dataset/audio"
MODEL_PATH = "model/audio_model.h5"

SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40

# =======================
# FEATURE EXTRACTION
# =======================
def extract_audio_features(file_path):
    """
    Extract MFCC features from audio file
    Output shape: (40,)
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        # Pad audio if too short
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))

        # MFCC extraction
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Mean pooling over time
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean.astype(np.float32)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# =======================
# LOAD DATASET
# =======================
X, y = [], []

print("\n📥 Loading audio dataset...")

for label, category in enumerate(["FAKE", "REAL"]):
    category_path = os.path.join(AUDIO_PATH, category)

    if not os.path.exists(category_path):
        print(f"⚠ Missing folder: {category_path}")
        continue

    print(f"Processing {category} audio files...")

    for file in os.listdir(category_path):
        if file.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(category_path, file)
            features = extract_audio_features(file_path)

            if features is not None and features.shape == (N_MFCC,):
                X.append(features)
                y.append(label)

print(f"\n✅ Total samples loaded: {len(X)}")

X = np.array(X)
y = np.array(y)

# =======================
# DATA CHECK
# =======================
print("\n📊 Label Distribution:")
print("FAKE:", np.sum(y == 0))
print("REAL:", np.sum(y == 1))

# =======================
# TRAIN-TEST SPLIT
# =======================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reshape for CNN → (samples, features, channels)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("\n🔍 Input shape:", X_train.shape)

# =======================
# MODEL ARCHITECTURE
# =======================
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu",
           input_shape=(N_MFCC, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =======================
# TRAIN MODEL
# =======================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\n🚀 Training Audio Authenticity Model...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop]
)

# =======================
# SAVE MODEL
# =======================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

print(f"\n✅ Model saved successfully at: {MODEL_PATH}")
