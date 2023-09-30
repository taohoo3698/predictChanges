import streamlit as st
import pandas as pd
import joblib

# Load the saved model
loaded_model = joblib.load('linear_regression_model.pkl')

# Define the Streamlit app
st.title('Insurance Charges Prediction')

# Input form for user to enter values
st.sidebar.header('Enter Your Information')
age = st.sidebar.slider('Age', 18, 100, 30)
bmi = st.sidebar.slider('BMI', 15.0, 50.0, 25.0)
children = st.sidebar.slider('Number of Children', 0, 5, 2)

# Create a DataFrame from user input
input_data = pd.DataFrame({'age': [age], 'bmi': [bmi], 'children': [children]})

# Make predictions when the user clicks the 'Predict' button
if st.sidebar.button('Predict'):
    predicted_charges = loaded_model.predict(input_data)
    st.success(f'Predicted Charges: ${predicted_charges[0]:.2f}')

# Display some information about the model
st.write('This app uses a linear regression model to predict insurance charges based on your input.')

# Optionally, you can provide additional information about the insurance dataset here.
# st.write('The model was trained on a dataset of insurance charges.')

# Optionally, provide a link to the source code or dataset.
# st.write('Source code and dataset: [GitHub Repo](https://github.com/yourusername/your-repo)')

