import streamlit as st

#Home
st.title('Home')

st.header('Welcome to CarVision!')

#About
st.header('About')

st.write(  
    """ 
    This web application is a **Car Recognition System** powered by a deep learning model based on **ResNet50**. 
    It is designed to accurately classify car images into **196 different car models**.  

    ### 🔍 How It Works  
    - Users upload an image of a car through the app.  
    - The image is preprocessed to match the required input format of the **ResNet50** model.  
    - The trained **TensorFlow Keras** model analyzes the image and predicts the most likely car model.  
    - The app then displays the top three options for the **predicted car model name** along with the **confidence score** as a percentage.  

    ### 🚀 Technology Stack  
    - **Deep Learning Model:** ResNet50 (pre-trained and fine-tuned on a car dataset)  
    - **Backend:** TensorFlow Keras for model inference  
    - **Frontend:** Streamlit for a user-friendly web interface  
    - **Deployment:** Hosted on streamlit's cloud platform  

    This app aims to provide an efficient and intuitive solution for identifying car models with high accuracy.  
    Try it out by uploading an image of a car! 🚗🔎  
    """
)


