import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os

# Add the directory containing the app module to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from predict import predict_car

# Streamlit code
st.title('Car Recognition')
st.write('This is a simple web app to recognize cars.')
st.write('Upload a car image and the model will predict the car make and model.')

# Create interface to interact with prediction model
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Starting prediction...")
    # Save the uploaded image to a temporary file
    temp_image_path = "car_recognition_project\temp\temp_image.jpg"
    image.save(temp_image_path)
    
    # Predict the car make and model
    top_predictions = predict_car(temp_image_path)
    st.write("Prediction completed.")
    
    if isinstance(top_predictions, str):
        st.write(top_predictions)
    else:
        for label, probability in top_predictions:
            st.write(f"Label: {label}, Probability: {probability:.2%}")