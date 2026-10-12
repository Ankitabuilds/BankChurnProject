
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the likelihood "
    "of the customer leaving the bank."
)

# Customer details form
with st.form("customer_form"):
    credit_score = st.number_input(
        "Credit Score", min_value=0, max_value=1000, value=650
    )
    geography = st.selectbox(
        "Geography", ["France", "Germany", "Spain"]
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=40
    )
    tenure = st.number_input(
        "Tenure (years)", min_value=0, max_value=10, value=3
    )
    balance = st.number_input(
        "Account Balance", min_value=0.0, value=60000.0
    )
    products = st.selectbox(
        "Number of Products", [1, 2, 3, 4]
    )
    credit_card = st.selectbox(
        "Has Credit Card?", ["Yes", "No"]
    )
    active_member = st.selectbox(
        "Is Active Member?", ["Yes", "No"]
    )
    salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1 if credit_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    processed_customer = preprocessor.transform(customer)
    prediction = model.predict(processed_customer)[0]
    probability = model.predict_proba(processed_customer)[0][1]

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.2%}")

    if prediction == 1:
        st.warning("The model predicts that this customer ma" \
        "y leave.")
    else:
        st.success("The model predicts that this customer may stay.")

    st.caption(
        "This is a machine learning estimate, not a guarantee "
        "of future customer behavior."
    )