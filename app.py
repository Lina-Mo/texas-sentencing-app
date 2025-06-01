import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load("xgboost_sentence_model.joblib")
scaler = joblib.load("age_scaler.joblib")
feature_columns = joblib.load("feature_columns_template.joblib")

st.set_page_config(page_title="Texas Sentence Estimator", layout="centered")
st.title("📏 Texas Sentence Length Estimator (XGBoost)")
st.write("Estimate the expected prison sentence based on user profile.")

# ---- USER INPUTS ----
age = st.slider("Age", min_value=17, max_value=90, value=30)
gender = st.selectbox("Gender", ["F", "M"])
race = st.selectbox("Race", ["White", "Black", "Hispanic", "Other and Unknown"])
county = st.selectbox("County", [
    "Harris", "Dallas", "Tarrant", "Bexar", "Travis", "El Paso", "Collin", "Hidalgo",
    "Lubbock", "Smith", "Cameron", "Fort Bend", "Montgomery", "Bell", "Jefferson"
])
offense = st.selectbox("Offense", [
    'Drug-Possession', 'Assault/Terroristic Threat/Trafficking', 'Burglary', 'DWI',
    'Robbery', 'Homicide', 'Forgery', 'Fraud', 'Weapons Offenses',
    'Obstruction/Public Order', 'Larceny', 'Stolen Vehicle', 'Other',
    'Sexual Assault', 'Drug-Delivery', 'Family Offense',
    'Sexual Assault Against a Child', 'Failure to Register as a Sex Offender',
    'Commercialized/Sex Offense', 'Kidnapping'
])

# ---- DATA PREPARATION ----
# Initialize input with zeros for all features
input_data = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

# Set age (after scaling)
scaled_age = scaler.transform([[age]])[0][0]
input_data.at[0, 'Age'] = scaled_age

# One-hot encode inputs if matching column exists
for col in [
    f'Gender_{gender}',
    f'Race_{race}',
    f'County_{county}',
    f'Offense_{offense}'
]:
    if col in input_data.columns:
        input_data.at[0, col] = 1

# ---- PREDICTION ----
if st.button("Estimate Sentence Length"):
    prediction = model.predict(input_data)[0]
    st.success(f"📌 Estimated Sentence Length: **{prediction:.2f} years**")
