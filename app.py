import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io

# Streamlit app title
st.title("Brain Tumor Segmentation with U-Net")

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
    # Find contours (for visualization, not used in model input)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    # Resize to target size
    processed_img = cv2.resize(dilated, target_size)
    # Convert back to 3 channels for U-Net
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    processed_img = processed_img / 255.0
    return processed_img, contour_img

# Create colored mask (red for tumor, green for non-tumor)
def create_colored_mask(prediction, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    # Threshold prediction to binary (tumor > 0.5)
    binary_mask = (prediction > 0.5).astype(np.uint8)
    # Create a 3-channel image for colored output
    colored_mask = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    # Set tumor regions to red (255, 0, 0)
    colored_mask[binary_mask == 1] = [255, 0, 0]
    # Set non-tumor regions to green (0, 255, 0)
    colored_mask[binary_mask == 0] = [0, 255, 0]
    return colored_mask

# Load pre-trained U-Net model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_unet_model.h5')
    return model

model = load_model()

# File uploaders
uploaded_image = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "png"])
uploaded_mask = st.file_uploader("Upload a ground truth mask (optional, JPG/PNG)", type=["jpg", "png"])

if uploaded_image is not None:
    # Read and preprocess image
    image = Image.open(uploaded_image)
    processed_img, contour_img = preprocess_image(image)
    
    # Display input and contour images
    st.image(image, caption="Input MRI Image", use_column_width=True)
    st.image(contour_img, caption="Image with Contours", use_column_width=True)
    
    # Perform prediction
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    prediction = model.predict(processed_img)[0]  # Remove batch dimension
    # Create colored mask
    colored_mask = create_colored_mask(prediction)
    
    # Display predicted colored mask
    st.image(colored_mask, caption="Predicted Tumor Mask (Red: Tumor, Green: No Tumor)", use_column_width=True)
    
    # Display ground truth mask if provided
    if uploaded_mask is not None:
        mask = Image.open(uploaded_mask).convert('L')  # Convert to grayscale
        mask = np.array(mask)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = (mask > 127).astype(np.uint8) * 255  # Threshold to binary
        st.image(mask, caption="Ground Truth Mask", use_column_width=True, clamp=True)

# Instructions
st.markdown("""
### Instructions
1. Upload an MRI image to segment the tumor region.
2. Optionally, upload a ground truth mask to compare with the prediction.
3. The app will display:
   - The input MRI image.
   - The image with detected contours.
   - The predicted tumor mask (red for tumor, green for no tumor).
   - The ground truth mask (if provided).
""")