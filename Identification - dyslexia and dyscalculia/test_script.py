import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("weakness_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to preprocess input and make a prediction
def predict_weakness(sinhala_exam, math_exam, sinhala_incorrect_ratio, math_incorrect_ratio):
    # Prepare the input data
    input_data = np.array([[sinhala_exam, math_exam, sinhala_incorrect_ratio, math_incorrect_ratio]])    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)  
    # Get the prediction
    prediction = model.predict(input_data_scaled)  
    # Map the prediction to the target class
    prediction_label = "Sinhala" if prediction[0] == 1 else "Math"  
    return prediction_label

    #incorrect ratio = (incorrect answer count / total answer count)*0.5

sinhala_exam = 90
math_exam = 75
sinhala_incorrect_ratio = 0.5
math_incorrect_ratio = 0.2

predicted_weakness = predict_weakness(sinhala_exam, math_exam, sinhala_incorrect_ratio, math_incorrect_ratio)
print(f"The predicted weakness is: {predicted_weakness}")
