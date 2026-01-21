import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/text_model.pkl")
vectorizer = joblib.load("model/text_vectorizer.pkl")

# -----------------------------
# TEST SAMPLES
# -----------------------------
samples = [
    "This essay discusses the impact of AI on education and society.",
    "Write a blog about space exploration in a poetic tone.",
    "The sun was setting behind the hills as she closed her notebook.",
    "Generate a professional LinkedIn post about AI startups."
]

X = vectorizer.transform(samples)
preds = model.predict(X)
probs = model.predict_proba(X)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
for text, pred, prob in zip(samples, preds, probs):
    label = "AI-GENERATED" if pred == 1 else "REAL (HUMAN)"
    confidence = prob[pred] * 100
    print(f"\n{label} ({confidence:.2f}%)")
    print(f"Text: {text}")
