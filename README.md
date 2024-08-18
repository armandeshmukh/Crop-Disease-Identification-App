# Crop Disease Detection

This project utilizes deep learning to detect crop diseases from images. The application enables farmers to upload images of their crops and receive predictions about potential diseases based on a trained model. The dataset used for training the model includes labeled images of healthy and diseased crops.

## Overview

The primary goal is to provide a user-friendly interface for farmers to identify diseases in their crops. The deep learning model, trained on various crop images, predicts the disease based on visual features. This tool aims to support farmers by providing timely and accurate disease identification.

## Setup
- Clone the repository: 'https://github.com/your-username/crop_disease_project.git'
- Navigate to the project directory: 'cd crop_disease_project'
- Set Up a Virtual Environment (Recommended): 'python -m venv venv source venv/bin/activate'   # On Windows use `venv\Scripts\activate`
- Install Dependencies: 'pip install -r requirements.txt'
- Train the Model (If Needed): If you need to train or update the model, ensure your dataset is correctly organized in the data/ directory and run: 'python train_model.py'
- Run the Streamlit App: 'streamlit run app.py'

## Usage 
- Upload an Image: Use the Streamlit interface to upload an image of your crop.
- View Predictions: The app will display the uploaded image along with the predicted disease.

## Results
- Disease Detection: The model will classify the uploaded image into one of the predefined categories (e.g., Healthy, Disease1, Disease2).
- Visualization: View the results directly in the Streamlit app, which provides real-time predictions based on the uploaded images.

## Contributors
- Arman Deshmukh

## Licenses 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

