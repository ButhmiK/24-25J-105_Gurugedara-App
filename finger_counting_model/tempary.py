import cv2
import tensorflow as tf
import numpy as np
import random
import time

# Load the HDF5 model
model = tf.keras.models.load_model("model/finger_counting_model.h5")

# Function to predict finger count based on the image
def predict_finger_count(image, model):
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to match model's input size (224x224)
    img_resized = cv2.resize(img_gray, (224, 224))  # Use 224x224 as expected by the model

    # Normalize the image as the model expects
    img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    img_resized = np.repeat(img_resized, 3, axis=-1)   # Convert to 3 channels (RGB)
    img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict finger count
    predictions = model.predict(img_resized)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_confidence = predictions[0][predicted_class]

    return predicted_class, predicted_confidence

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

frame_counter = 0  # To control prediction interval
last_random_number_time = time.time()  # Track time for random number display
random_number = random.randint(0, 3)  # Initial random number

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the random number on the screen for the user to show with fingers
    cv2.putText(frame, f"Show Number{random_number} ", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show prediction every 1 frame for smooth feedback
    predicted_class, confidence = predict_finger_count(frame, model)

    # Display predicted number of fingers
    cv2.putText(frame, f"Your Answer: {predicted_class}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display feedback ("Right" or "Wrong") based on comparison
    if predicted_class == random_number:
        feedback = "Right"
        feedback_color = (0, 255, 0)  # Green for correct
    else:
        feedback = "Wrong"
        feedback_color = (0, 0, 255)  # Red for incorrect

    # Display feedback ("Right" or "Wrong")
    cv2.putText(frame, feedback, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

    # Check if 5 seconds have passed since the last random number
    if time.time() - last_random_number_time > 5:
        random_number = random.randint(0, 5)  # Update the random number
        last_random_number_time = time.time()  # Reset timer

    # Show the frame with the random number, prediction, and feedback
    cv2.imshow("Finger Counting", frame)

    # Wait for key press, exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
