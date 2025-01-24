import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("random_forest_model.pkl", "rb"))

st.title("Bank account ownership Prediction in Africa")
st.header("Input individual Information")

# Selected features : gender_of_respondent, age_of_respondent, education_level and job_type
# Test with 1,40.0,0,5

gender = st.number_input("gender_of_respondent")
age = st.number_input("age_of_respondent")
education = st.number_input("education_level")
job = st.number_input("job_type")

# Prediction
if st.button("Predict"):
    input_data = np.array([[gender, age, education, job]])
    prediction = model.predict(input_data)
    st.write(f"Prediction: {'Has a bank account' if prediction[0] == 1 else 'Does not have a Bank account'}")
