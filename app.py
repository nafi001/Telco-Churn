import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_smote_model.joblib")

# Define function to preprocess user input
def preprocess_input(data):
    d_dict = {"tenure": [data["tenure"]], "MonthlyCharges": [data["MonthlyCharges"]], "TotalCharges": [data["TotalCharges"]]}
    
    categorical_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod"]
    
    for col in categorical_cols:
        for value in ["Female", "Male", "0", "1", "No", "Yes", "DSL", "Fiber optic", "No internet service", 
                      "Month-to-month", "One year", "Two year", "Bank transfer (automatic)", "Credit card (automatic)", 
                      "Electronic check", "Mailed check"]:
            d_dict[f"{col}_{value}"] = [1 if data[col] == value else 0]
    
    return pd.DataFrame.from_dict(d_dict)

# Define prediction function
def predict_churn(input_data):
    df = preprocess_input(input_data)
    churned = model.predict(df)[0]
    prediction = model.predict_proba(df)[:, 1][0]
    pred_percent = round(prediction * 100, 2)
    
    outcome = "This customer is not likely to churn" if churned == 0 else "This customer is likely to churn"
    
    if pred_percent >= 85:
        risk = "Very High"
    elif pred_percent >= 75:
        risk = "High"
    elif pred_percent >= 50:
        risk = "Moderate"
    elif pred_percent >= 25:
        risk = "Low"
    else:
        risk = "Very Low"
    
    return outcome, f"Probability of Churning: {pred_percent}%", f"Risk Level: {risk}"

# Streamlit UI
st.title("Customer Churn Prediction App")

st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["0", "1"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

if st.sidebar.button("Predict Churn"):
    input_data = {
        "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone_service, "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method
    }
    
    result, confidence, risk = predict_churn(input_data)
    st.subheader("Prediction Result")
    st.write(result)
    st.write(confidence)
    st.write(risk)
