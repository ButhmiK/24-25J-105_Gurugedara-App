import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the HDF5 model
model = tf.keras.models.load_model("model/finger_counting_model.h5")

def predict_finger_count(image_path, model):
    # Load the image, convert to grayscale, and resize it
    img = Image.open(image_path).convert("L").resize((224, 224))

    # Convert grayscale to a NumPy array for OpenCV processing
    img_array = np.array(img)

    # Apply Gaussian Blur to reduce noise
    img_blur = cv2.GaussianBlur(img_array, (5, 5), 0)

    # Apply thresholding to create a binary mask
    _, binary_mask = cv2.threshold(img_blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert binary mask back to RGB by stacking channels
    binary_rgb = np.stack([binary_mask]*3, axis=-1)

    # Resize and preprocess the image for MobileNetV2
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(binary_rgb)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    # Get predictions
    predictions = model.predict(img_preprocessed)

    # Get the class with the highest score
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_confidence = predictions[0][predicted_class]

    return predicted_class, predicted_confidence

# Folder path containing the images
folder_path = 'test_images'
# Supported image extensions
valid_extensions = ('.jpg', '.jpeg', '.png')
# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(valid_extensions):  # Check if file has a valid image extension
        image_path = os.path.join(folder_path, filename)
        predicted_class, confidence = predict_finger_count(image_path, model)
        print(f"File Name: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}")
        
        # Open and display the image
        with Image.open(image_path) as img:
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"File Name: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}")
            plt.show()
