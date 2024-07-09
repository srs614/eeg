import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input to pandas DataFrame with appropriate column names
    columns = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
    data = pd.DataFrame([data], columns=columns)
    # Scale the data using the loaded scaler
    data = scaler.transform(data)
    # Reshape data for LSTM
    data = data.reshape(data.shape[0], data.shape[1], 1)
    return data

# Streamlit app interface
st.title("EEG Confusion Detection using BiLSTM")

# Create input fields for raw values in two columns
col1, col2 = st.columns(2)

with col1:
    delta = st.number_input('Delta')
    alpha1 = st.number_input('Alpha1')
    beta1 = st.number_input('Beta1')
    gamma1 = st.number_input('Gamma1')

with col2:
    theta = st.number_input('Theta')
    alpha2 = st.number_input('Alpha2')
    beta2 = st.number_input('Beta2')
    gamma2 = st.number_input('Gamma2')

# Button for prediction
if st.button('Predict'):
    # Collect input data
    input_data = [delta, theta, alpha1, alpha2, beta1, beta2, gamma1, gamma2]
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    # Make prediction
    prediction = model.predict(processed_data)
    # Extract probability from prediction
    probability = prediction[0][0]
    # Convert prediction to label
    label = 'Confused' if probability > 0.5 else 'Not Confused'
    # Display the result
    st.write(f'The model predicts that the student is: {label}')
