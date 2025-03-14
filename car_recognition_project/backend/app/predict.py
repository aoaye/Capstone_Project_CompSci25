import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_loader import model
from app.image_preprocessing import preprocess_image

# Load labels from the existing CSV file
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # Extract unique label names and ensure they are correctly indexed starting from 1
    label_names = df['label_names'].apply(lambda x: x.strip("[]'")).unique()
    label_names = {i+1:label for i, label in enumerate(label_names)}
    return label_names

# Update with the path to your existing CSV file
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'cars_annos.csv'))
label_names = load_labels(csv_path)

def predict_car(image_path, top_n=3, threshold=0.1):
    print("Preprocessing image...")
    image = preprocess_image(image_path)
    print("Image preprocessed. Making predictions...")
    predictions = model.predict(image)
    print("Predictions made. Processing results...")
    
    # Ensure predictions are correctly indexed
    top_indices = np.argsort(predictions[0])[::-1][:top_n]
    top_labels = [(label_names[i], predictions[0][i]) for i in top_indices]
    
    # Check if the highest probability is below the threshold
    if top_labels[0][1] < threshold:
        return "No similarity found"
    
    return top_labels
