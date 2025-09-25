import streamlit as st
import numpy as np
import joblib

model = joblib.load("poly3_regression_model.pkl")
poly = joblib.load("poly3_transformer.pkl")
scaler_reg = joblib.load("prepared_data/standard_scaler_reg.pkl")

st.set_page_config(page_title="Total Score Prediction App", layout="centered")
st.title("Total Score Prediction (Polynomial Regression, Degree 3)")
st.write("Enter the student's weekly self-study hours to predict the total score:")

weekly_self_study_hours = st.number_input("Weekly Self Study Hours", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
weekly_self_study_hours = min(weekly_self_study_hours,  24.10)
if st.button("Predict Total Score"):
    X_input_raw = np.array([[weekly_self_study_hours]])
    # Scale input using regression scaler
    X_input_scaled = scaler_reg.transform(X_input_raw)
    X_poly = poly.transform(X_input_scaled)
    total_score_pred = model.predict(X_poly)
    st.success(f"Predicted Total Score: {total_score_pred[0]:.2f}")

st.info("Input is raw (not scaled). The app will scale and transform it using the saved regression StandardScaler and PolynomialFeatures before prediction.")
