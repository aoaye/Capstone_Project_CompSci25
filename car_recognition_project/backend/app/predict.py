from app.model_loader import model
from app.image_preprocessing import preprocess_image
import numpy as np

CLASS_LABELS = ["car_model_1", "car_model_2", "car_model_3"]  # Update with actual labels

def predict_car(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CLASS_LABELS[predicted_class]
