import joblib

model = joblib.load("model/text_model.pkl")
vectorizer = joblib.load("model/text_vectorizer.pkl")

def predict_text(text):
    vector = vectorizer.transform([text])
    pred = model.predict(vector)[0]
    return "🟢 REAL Text" if pred == 1 else "🔴 AI-GENERATED Text"
