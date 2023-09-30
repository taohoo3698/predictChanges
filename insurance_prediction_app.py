import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the saved model
loaded_model = joblib.load('linear_regression_model.pkl')

# Define the features for input
features = ['age', 'bmi', 'children']

# Define the Streamlit app
st.title("Insurance Charges Prediction")

# Create a sidebar for user input
st.sidebar.header('Enter Example Data')

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30)
bmi = st.sidebar.slider("BMI", min_value=15, max_value=50, value=25)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=2)

# Create a DataFrame with the user input
example_data = pd.DataFrame([[age, bmi, children]], columns=features)

# Make predictions
predicted_charges = loaded_model.predict(example_data)

# Display the prediction
st.sidebar.header('Prediction')
st.sidebar.write(f"Predicted Charges: ${predicted_charges[0]:.2f}")

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
actual_charges = 10000  # Replace with the actual charges if available

if actual_charges:
    mse = mean_squared_error([actual_charges], [predicted_charges[0]])
    rmse = np.sqrt(mse)
    st.sidebar.header('Model Evaluation Metrics')
    st.sidebar.write(f"MSE: {mse:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")
else:
    st.sidebar.write("Actual charges not provided. Unable to calculate MSE and RMSE.")

# Main content area
st.header('Insurance Charges Prediction')

# Display additional content or charts in the main area if needed
