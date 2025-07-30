import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io

# Streamlit app title
st.title("Brain Tumor Classification with CNN")

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Preprocessing function
def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    # Convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # Erode and dilate
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    # Find contours (for visualization)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    # Resize to target size (use original image for classification)
    processed_img = cv2.resize(img, target_size)
    processed_img = processed_img / 255.0
    return processed_img, contour_img

# Load pre-trained CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/workspaces/Predict-Brain-Tumor-CNN/brain_tumor_cnn_model.h5')
    return model

model = load_model()

# File uploader
uploaded_image = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "png"])

if uploaded_image is not None:
    # Read and preprocess image
    image = Image.open(uploaded_image)
    processed_img, contour_img = preprocess_image(image)
    
    # Display input and contour images
    st.image(image, caption="Input MRI Image", use_column_width=True)
    st.image(contour_img, caption="Image with Contours", use_column_width=True)
    
    # Perform prediction
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    prediction = model.predict(processed_img)[0]  # Get probability
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    text_color = "#FF0000" if prediction > 0.5 else "#00FF00"  # Red for tumor, green for no tumor
    #color = [255, 0, 0] if prediction > 0.5 else [0, 255, 0]  # Red for tumor, green for no tumor
    
    # Create colored background
    #colored_bg = np.zeros((100, 200, 3), dtype=np.uint8)
    #colored_bg[:] = 
    #st.image(colored_bg, caption=f"Prediction: {label}", use_column_width=True)
    
    # Display prediction with colored text
    st.markdown(f"<h3 style='color:{text_color};'>Prediction: {label}</h3>", unsafe_allow_html=True)
        
    # Display confidence with colored text
    st.markdown(f"<p style='color:{text_color};'>Prediction Confidence: {prediction[0]:.2f}</p>", unsafe_allow_html=True)

# Instructions
st.markdown("""
### Instructions
1. Upload an MRI image to classify as tumor or no tumor.
2. The app will display:
   - The input MRI image.
   - The image with detected contours.
   - The prediction (red for Tumor, green for No Tumor).
""")
