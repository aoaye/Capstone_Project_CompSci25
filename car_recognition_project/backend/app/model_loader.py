from tensorflow.keras.models import load_model
import os
import sys

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50.keras'))

def load_trained_model():
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model

# Load model on startup
model = load_trained_model()