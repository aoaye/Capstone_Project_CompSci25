from tensorflow.keras.models import load_model

MODEL_PATH = "models/resnet50_model.keras"

def load_trained_model():
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model

# Load model on startup
model = load_trained_model()