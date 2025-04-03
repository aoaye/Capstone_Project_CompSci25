# Car Recognition App

## Overview
This project is a deep learning-based car recognition application that allows users to identify cars from images. The model classifies cars into their respective make, model, and year. The application was built using TensorFlow Keras and ResNet50 for image classification and is deployed via a web-based Streamlit application and an API backend.

## Features
- Upload an image of a car to get its make, model, and year.
- Uses a deep learning model (ResNet50) trained on the Stanford Cars dataset.
- API backend for handling predictions.
- Streamlit-based web application for an easy-to-use interface.
- Flutter mobile application for most convenient usage.
- Deployed to AWS ECR and AppRunner for scalability.

## Tech Stack
### **Machine Learning**
- TensorFlow Keras (ResNet50, VGG19)
- NumPy & Pandas (data processing)

### **Backend**
- FastAPI (for API development)
- Uvicorn (for running the server)
- Docker (containerization)
- AWS ECR & AppRunner (deployment)

### **Frontend**
- Streamlit (for web UI)
- Flutter (for mobile app)

## Installation

### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- FastAPI
- Streamlit
- Docker (if running in a container)

### **Setup**
Clone the repository:
```sh
git clone https://github.com/aoaye/capstone_project_compsci25.git
cd car-recognition-app
```

### **Install dependencies**
```sh
pip install -r requirements.txt
```

### **Running the Web App**
```sh
streamlit run streamlit_app.py
```

### **Running the API**
```sh
uvicorn backend.routes:app --host 0.0.0.0 --port 8000
```

## Model Training
The model was trained using the Stanford Cars dataset. The dataset was preprocessed and split into training, validation, and test sets. ResNet50 was fine-tuned to classify 196 different car classes.

## Deployment
The API and web app are deployed using AWS services:
- **Model**: Packaged in a Docker container and pushed to AWS ECR.
- **API**: Hosted on AWS AppRunner for scalability.

## Future Improvements
- Improve model accuracy with additional data augmentation.
- Optimize API response times.
- Add Prediction History feature

## Contributing
Feel free to submit issues or pull requests to improve the project.
