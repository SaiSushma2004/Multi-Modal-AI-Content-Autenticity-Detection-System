import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
DATA_PATH = "dataset/text/ai_vs_human_text.csv"
df = pd.read_csv(DATA_PATH)

# ---- STEP 1: Normalize labels ----
df["label"] = df["label"].str.lower().str.strip()

# ---- STEP 2: Map labels to numbers ----
label_map = {
    "ai-generated": 1,
    "ai generated": 1,
    "gpt": 1,
    "fake": 1,

    "human-written": 0,
    "human": 0,
    "real": 0
}

df["label"] = df["label"].map(label_map)

# ---- STEP 3: Drop invalid rows ----
df = df.dropna(subset=["label"])

# ---- STEP 4: Features & labels ----
X = df["text"]
y = df["label"].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF with better detection power
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 3)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

# Train
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/text_model.pkl")
joblib.dump(vectorizer, "model/text_vectorizer.pkl")

print("✅ Text model trained and saved successfully!")
