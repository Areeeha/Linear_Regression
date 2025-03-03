import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Title
st.title("Employee Salary Prediction - With Custom Dataset Support & Visualizations")

# File uploader - allows user to upload custom dataset
uploaded_file = st.file_uploader("Upload your CSV file (with columns: Job Title, Years of Experience, Education Level, Age, Salary)", type="csv")

# Define function to visualize dataset trends
def show_visualizations(data, title_suffix=""):
    st.write(f"### Salary vs Years of Experience {title_suffix}")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Years of Experience', y='Salary', data=data, ax=ax)
    st.pyplot(fig)

    st.write(f"### Salary Distribution by Top Job Titles {title_suffix}")
    top_titles = data['Job Title'].value_counts().head(15).index
    data_top = data[data['Job Title'].isin(top_titles)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Job Title', y='Salary', data=data_top, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# If user uploads a file, use that data
if uploaded_file is not None:
    st.success("Custom dataset uploaded successfully. Retraining model...")

    # Load and process uploaded dataset
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['Salary'])
    df['Years of Experience'] = df['Years of Experience'].fillna(df['Years of Experience'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Job Title'] = df['Job Title'].fillna(df['Job Title'].mode()[0])
    df['Education Level'] = df['Education Level'].fillna(df['Education Level'].mode()[0])

    # Show uploaded data preview
    st.write("### Uploaded Dataset Preview")
    st.write(df.head())

    # Visualize uploaded data (Bonus)
    show_visualizations(df, "(Uploaded Data)")

    # Set features and target
    X = df[['Job Title', 'Years of Experience', 'Education Level', 'Age']]
    y = df['Salary']

    # Preprocessing pipeline
    numeric_features = ['Years of Experience', 'Age']
    categorical_features = ['Job Title', 'Education Level']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Train model on uploaded data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"### Model Retrained on Uploaded Data - MSE: {mse:.2f}")

    # Extract dropdown options from uploaded data
    job_titles = sorted(list(df['Job Title'].unique()))
    education_levels = sorted(list(df['Education Level'].unique()))

else:
    # Load pre-trained model & mappings if no file is uploaded
    st.info("Using pre-trained model (linear_model.pkl) — upload a dataset to retrain.")

    with open('linear_model.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    with open('mappings.json', 'r') as file:
        mappings = json.load(file)

    job_titles = mappings['job_titles']
    education_levels = mappings['education_levels']

    # Load pre-trained dataset for visualization
    df = pd.read_csv('Salary Data.csv')

    # Visualize pre-trained dataset trends (Bonus - Always-On Visualization)
    show_visualizations(df, "(Pre-trained Data)")

# --- Salary Prediction Section ---
st.write("## Predict Salary")

job_title = st.selectbox("Select Job Title", job_titles)
experience_years = st.number_input("Enter Years of Experience", min_value=0, max_value=50, value=5)
education_level = st.selectbox("Select Education Level", education_levels)
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)

if st.button("Predict Salary"):
    input_data = pd.DataFrame([[job_title, experience_years, education_level, age]],
                              columns=['Job Title', 'Years of Experience', 'Education Level', 'Age'])

    predicted_salary = pipeline.predict(input_data)[0]
    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')

# Footer
st.sidebar.info("Upload your own dataset to retrain the model or use the pre-trained model.")
