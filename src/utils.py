"# utils.py" 
import joblib
import json
import os
from datetime import datetime

def save_model(model, vectorizer=None, path="models", name="model"):
    """Save model and optional vectorizer."""
    os.makedirs(path, exist_ok=True)
    model_path = f"{path}/{name}.pkl"
    joblib.dump(model, model_path)

    if vectorizer:
        vectorizer_path = f"{path}/{name}_vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)

    return model_path

def load_model(path, vectorizer=False):
    """Load saved model and vectorizer."""
    model = joblib.load(path)
    if vectorizer:
        v_path = path.replace(".pkl", "_vectorizer.pkl")
        vect = joblib.load(v_path)
        return model, vect
    return model

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def print_box(message):
    print("\n" + "="*50)
    print(message)
    print("="*50 + "\n")
