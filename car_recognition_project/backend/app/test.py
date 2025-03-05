import sys
import os
from PIL import Image

# Add the directory containing the app module to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.predict import predict_car

# Example usage
image_path = r"temp\AMV12.jpg"
print("Starting prediction...")
top_predictions = predict_car(image_path)
print("Prediction completed.")
if isinstance(top_predictions, str):
    print(top_predictions)
else:
    for label, probability in top_predictions:
        print(f"Label: {label}, Probability: {probability:.2%}")