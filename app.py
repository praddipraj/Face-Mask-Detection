import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("mask_detection_model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    return image_array

# Function to make predictions
def predict_mask(image):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    if prediction[0][0] < 0.5:
        result = "with_mask"
    else:
        result = "without_mask"
    return result

# Streamlit app
def main():
    st.title("Face Mask Detection")
    st.write("Upload an image to check if the person is wearing a mask or not.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict if the person is wearing a mask or not
        result = predict_mask(image)
        st.write("Prediction:", result)

if __name__ == "__main__":
    main()
