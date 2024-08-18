import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model (adjust the path as needed)
model = tf.keras.models.load_model('crop_disease_model.h5')

def predict_disease(img):
    img = img.resize((224, 224))  # Adjust based on your model's expected input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = ['Healthy', 'Disease1', 'Disease2']  # Update with your actual class names
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class

def main():
    st.title("Crop Disease Detection")

    st.sidebar.header("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        prediction = predict_disease(image)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
