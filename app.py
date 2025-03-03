import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

# Load trained pipeline
with open('linear_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Load dropdown options (sorted order already applied in Colab)
with open('mappings.json', 'r') as file:
    mappings = json.load(file)

job_titles = mappings['job_titles']
education_levels = mappings['education_levels']

# Streamlit App
st.title("Employee Salary Prediction")
st.write("This app predicts employee salary based on Job Title, Years of Experience, Education Level, and Age.")

# User Inputs
job_title = st.selectbox("Select Job Title", job_titles)
experience_years = st.number_input("Enter Years of Experience", min_value=0, max_value=50, value=5)
education_level = st.selectbox("Select Education Level", education_levels)
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)

# Predict Salary
if st.button("Predict Salary"):
    input_data = pd.DataFrame([[job_title, experience_years, education_level, age]],
                              columns=['Job Title', 'Years of Experience', 'Education Level', 'Age'])

    predicted_salary = pipeline.predict(input_data)[0]
    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
