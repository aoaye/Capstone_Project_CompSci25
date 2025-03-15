import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import requests

# Streamlit code
st.title('CarVision')
st.write('This is a simple web app to recognize cars.')
st.write('Upload a car image and the model will predict the car make and model.')

# Create interface to interact with prediction model
uploaded_file = st.file_uploader("Choose an image...", type="jpg, jpeg, png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Starting prediction...")
    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    
    # Predict the car make and model using AWS App Runner server
    url = "https://jsmrkvzrep.eu-west-1.awsapprunner.com/predict/"
    with open(temp_image_path, "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
    
    if response.status_code == 200:
        top_predictions = response.json()["prediction"]
        st.write("Prediction completed.")
        for prediction in top_predictions:
            label = prediction["label"]
            probability = prediction["probability"]
            st.write(f"Label: {label}, Probability: {probability:.2%}")
    else:
        st.write("Failed to get prediction. Status code:", response.status_code)