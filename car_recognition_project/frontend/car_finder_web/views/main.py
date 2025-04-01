import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout, RequestException
from stqdm import stqdm
from time import sleep

st.title('CarVision')
st.write(
    """
    Upload an image of a car, and our model will identify its make and model with high accuracy. 
    Make sure the image is clear and well-lit for the best results!
    """
)

# Create interface to interact with prediction model
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Starting prediction...")
    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    
    # Predict the car make and model using AWS App Runner server
    url = "https://jsmrkvzrep.eu-west-1.awsapprunner.com/predict/"
    try:
        with open(temp_image_path, "rb") as image_file:
            response = requests.post(url, files={"file": image_file}, timeout=10)
            
            # Simulate waiting for the prediction to complete, for demonstration purposes
            # while actual prediction time is much shorter
            for _ in stqdm(range(10), "Waiting for prediction..."):
                sleep(0.5)
        
        if response.status_code == 200:
            top_predictions = response.json()["prediction"]
            st.write("Prediction completed.")
            
            with st.expander("Top Predictions"):
                for prediction in top_predictions:
                    label = prediction["label"]
                    probability = prediction["probability"]
                    st.write(f"We're **{probability:.2%}** sure it is a **{label}**.")
        else:
            st.write("Failed to get prediction. Status code:", response.status_code)
    
    except ConnectionError:
        st.error("No connection: Unable to reach the server. Please check your internet connection or try again later.")
    except Timeout:
        st.error("Request timed out: The server took too long to respond. Please try again later.")
    except HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except RequestException as req_err:
        st.error(f"An error occurred: {req_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")