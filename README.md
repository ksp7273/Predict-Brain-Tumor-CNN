Collecting workspace informationHere’s a step-by-step explanation of how your notebook brain_tumor_segmentation_unet.ipynb works for brain tumor classification using a CNN:

---

### 1. **Introduction & Dependencies**
- The notebook starts with a markdown cell explaining the project: binary classification of MRI images as "tumor" or "no tumor".
- Installs required libraries: TensorFlow, NumPy, Matplotlib, OpenCV, Scikit-learn, Streamlit, Pillow.

---

### 2. **Import Libraries**
- Imports all necessary Python libraries for deep learning, image processing, and plotting.

---

### 3. **Dataset Paths & Parameters**
- Sets up paths to the dataset:
  - Images are in yes and no.
- Defines image size (`128x128`) and batch size.

---

### 4. **Exploratory Data Analysis (EDA)**
- Lists all image files in the "yes" and "no" folders.
- Creates labels: `1` for tumor, `0` for no tumor.
- Prints the number of images in each class and checks the shape of a sample image.

---

### 5. **Plot Sample Images**
- Plots a few sample images from both classes to visually inspect the data.

---

### 6. **Data Augmentation (Function Defined)**
- Defines a function to create data generators with optional augmentation using Keras’ `ImageDataGenerator`.
- (Note: In this notebook, this function is defined but not used in the main training pipeline.)

---

### 7. **Image Preprocessing**
- Defines a function to preprocess each image:
  - Reads the image.
  - Converts to grayscale.
  - Applies Gaussian blur and thresholding.
  - Erodes and dilates to highlight features.
  - Finds and draws contours (for visualization).
  - Resizes and normalizes the image for model input.

---

### 8. **Load and Preprocess Images**
- Loads all images, applies the preprocessing function, and stores them in a NumPy array.
- Converts labels to a NumPy array.
- Prints the shapes of the loaded images and labels.

---

### 9. **Train/Validation/Test Split**
- Splits the dataset into:
  - 70% training
  - 15% validation
  - 15% test
- Prints the number of images in each split.

---

### 10. **CNN Model Definition & Training**
- Defines a simple CNN architecture:
  - 3 convolutional layers with increasing filters (32, 64, 128), each followed by max pooling.
  - Flatten, dense layer (128 units), dropout, and final sigmoid output.
- Compiles the model with Adam optimizer and binary cross-entropy loss.
- Prints the model summary.
- Trains the model on the training set, validating on the validation set for 20 epochs.

---

### 11. **Evaluation & Visualization**
- Evaluates the trained model on the test set and prints test accuracy.
- Saves the trained model as brain_tumor_cnn_model.h5 (used by `app.py`).
- Plots training/validation accuracy and loss curves.
- Shows predictions for a few test images, displaying both predicted and true labels.

---

## **Summary of Workflow**
1. **Setup**: Install libraries, import modules, set parameters.
2. **Data Loading**: Read image files, assign labels.
3. **EDA**: Visualize data distribution and sample images.
4. **Preprocessing**: Prepare images for model input.
5. **Splitting**: Divide data into train/val/test sets.
6. **Modeling**: Build, compile, and train a CNN.
7. **Evaluation**: Test model, visualize performance, and save the model.

---
This notebook prepares a CNN model for brain tumor classification, saves it as brain_tumor_cnn_model.h5, and provides visualizations for both the data and the model’s performance. The saved model can then be used in your Streamlit app (`app.py`) for real-time predictions.



Here’s a step-by-step explanation of how your app.py Streamlit code works for brain tumor classification:

---

### 1. **Import Libraries**
- Imports Streamlit for the web app, TensorFlow for loading the trained model, OpenCV and NumPy for image processing, and PIL for image handling.
         
---

### 2. **Set App Title**
- Displays the app title: **"Brain Tumor Classification with CNN"**.

---

### 3. **Define Image Parameters**
- Sets the image size to 128x128 pixels (must match the model’s input size).

---

### 4. **Preprocessing Function**
- `preprocess_image`:
  - Converts the uploaded image to a NumPy array.
  - Converts it to grayscale, applies Gaussian blur, thresholding, erosion, and dilation.
  - Finds and draws contours (for visualization).
  - Resizes the image and normalizes pixel values for model input.
  - Returns both the processed image (for prediction) and the contour image (for display).

---

### 5. **Load the Trained Model**
- load the trained CNN model (brain_tumor_cnn_model.h5) only once for efficiency.

---

### 6. **Image Upload**
- Provides a file uploader for the user to upload an MRI image (JPG/PNG).

---

### 7. **Image Processing and Prediction**
- If an image is uploaded:
  - Reads and preprocesses the image.
  - Displays the original and contour images.
  - Expands dimensions to match model input shape.
  - Runs the model to get a prediction (probability).
  - Sets the label to "Tumor" if probability > 0.5, else "No Tumor".
  - Sets text color: red for tumor, green for no tumor.
  - Displays the prediction and confidence with colored text.

---

### 8. **Instructions**
- Shows user instructions on how to use the app.

---

**Install**:  
        
         ```bash
          pip install streamlit tensorflow numpy opencv-python pillow


**RUN APP**: 
        
         ```bash
          streamlit run app.py

**Summary:**  
The app lets users upload an MRI image, processes it, shows the original and contour-highlighted images, predicts if a tumor is present, and displays the result with color-coded confidence.




