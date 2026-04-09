import streamlit as st
import numpy as np
import joblib

# -------------------------------
# PAGE SETTINGS (MUST BE FIRST)
# -------------------------------
st.set_page_config(page_title="Gastric Predictor", layout="centered")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("outputs/models/best_model.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.title("Gastric Disease Prediction System")
st.write("Fill patient details:")

# -------------------------------
# INPUTS I'M GIVING TO YOU
# -------------------------------
age = st.slider("Age", 15, 80)
gender = st.selectbox("Gender", ["Female", "Male"])
bmi = st.number_input("BMI", 15.0, 40.0)

smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol", ["None", "Moderate", "Heavy"])
family = st.selectbox("Family History", ["No", "Yes"])

stress = st.slider("Stress Level (0-10)", 0, 10)
spicy = st.slider("Spicy Food Intake (0-10)", 0, 10)

# -------------------------------
# CONVERT INPUTS
# -------------------------------
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
family = 1 if family == "Yes" else 0
alcohol = ["None", "Moderate", "Heavy"].index(alcohol)

# -------------------------------
# CREATE FEATURE ARRAY
# -------------------------------
features = np.array([[age, gender, bmi, smoking, alcohol, family,
                      spicy, stress, 0, 0, 0, 0, 0, 0]])

# Scale
features_scaled = scaler.transform(features)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk immediately diagonose it ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk with good health({probability:.2%})")