import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open('classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App title
st.title("Income Prediction")

# User input fields
age = st.number_input("Age", min_value=1, max_value=100, value=30)
education_num = st.number_input("Education Number (e.g. Bachelors = 13)", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", min_value=0.0, value=0.0)
capital_loss = st.number_input("Capital Loss", min_value=0.0, value=0.0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

married = st.radio("Are you married?", ['Yes', 'No'])
occupation_sales = st.radio("Do you work in sales?", ['Yes', 'No'])

# Convert categorical inputs
married_val = 1 if married == 'Yes' else 0
occupation_val = 1 if occupation_sales == 'Yes' else 0

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'age': age,
        'education-num': education_num,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'marital-status_Married-civ-spouse': married_val,
        'occupation_Sales': occupation_val
    }

    input_df = pd.DataFrame([input_data])

    # Apply scaling and log transformation
    num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])
    input_df['age'] = np.log1p(input_df['age'])

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 0:
        st.success("Predicted Income: Less than 50K")
    else:
        st.success("Predicted Income: More than 50K")
