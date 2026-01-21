import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# CONFIG
# -----------------------
AUDIO_PATH = "dataset/audio"
SAMPLE_RATE = 22050
DURATION = 3   # seconds
SAMPLES = SAMPLE_RATE * DURATION
MODEL_PATH = "model/audio_model.h5"

# -----------------------
# AUDIO FEATURE FUNCTION
# -----------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLES:
            padding = SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -----------------------
# LOAD DATASET
# -----------------------
X = []
y = []

print("\n📥 Loading audio dataset...")

for label, category in enumerate(["FAKE", "REAL"]):
    folder_path = os.path.join(AUDIO_PATH, category)

    if not os.path.exists(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        continue

    print(f"Processing {category} files...")

    for file in os.listdir(folder_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label)

print(f"\nTotal samples: {len(X)}")

X = np.array(X)
y = np.array(y)

# -----------------------
# CHECK DATA BALANCE
# -----------------------
print("\n📊 Label distribution:")
print("FAKE:", np.sum(y == 0))
print("REAL:", np.sum(y == 1))

# -----------------------
# TRAIN-TEST SPLIT
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# -----------------------
# BUILD MODEL
# -----------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# TRAIN MODEL
# -----------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n🚀 Training audio model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop]
)

# -----------------------
# SAVE MODEL
# -----------------------
model.save(MODEL_PATH)
print(f"\n✅ Audio model saved at: {MODEL_PATH}")
