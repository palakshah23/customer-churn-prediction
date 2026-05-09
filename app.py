import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

# Title
st.title("Customer Churn Prediction System")

st.write("Fill customer details below")

# Inputs
tenure = st.slider("Tenure (Months)", 1, 72)

monthly_charges = st.number_input("Monthly Charges")

total_charges = st.number_input("Total Charges")

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

contract = st.selectbox(
    "Contract Type",
    [0, 1, 2]
)

paperless = st.selectbox(
    "Paperless Billing",
    [0, 1]
)

# Prediction
if st.button("Predict Churn"):

    input_data = np.array([[
        tenure,
        monthly_charges,
        total_charges,
        senior,
        contract,
        paperless
    ]])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")

    st.write(
        "Prediction Probability:",
        probability
    )