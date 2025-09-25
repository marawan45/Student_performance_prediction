import streamlit as st
import numpy as np
import joblib
import pandas as pd


svm_model = joblib.load("best_logistic_regression_model.pkl")
scaler = joblib.load("prepared_data/standard_scaler.pkl")

st.set_page_config(page_title="Grade Prediction App", layout="centered")
st.title("Student Grade Prediction (SVM Model)")
st.write("Enter the student's features to predict the grade:")

# Create two columns: main and side
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("**Grade Mapping:**")
    st.markdown("""
    | Numeric | grade  |
    |---------|--------|
    |   0     |   A    |
    |   1     |   B    |
    |   2     |   C    |
    |   3     |   D    |
    |   4     |   F    |
    """)

with col1:
    weekly_self_study_hours = st.number_input("Weekly Self Study Hours", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    class_participation = st.number_input("Class Participation", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.button("Predict Grade"):
        X_input_raw_full = np.array([[weekly_self_study_hours, attendance_percentage, class_participation]])
        X_input_scaled_full = scaler.transform(X_input_raw_full)
        X_input_scaled = X_input_scaled_full
        pred = svm_model.predict(X_input_scaled)
        st.success(f"Predicted Grade: {pred[0]}")

st.info("Input values are raw (not scaled). The app will scale them using the saved StandardScaler before prediction.")
