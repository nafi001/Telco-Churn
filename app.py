import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
# Load the trained model
model = joblib.load('random_forest_smote_model.joblib')

def get_data():
    """Function to retrieve user input from Streamlit UI."""
    st.sidebar.header("Customer Information")
    tenure = st.sidebar.number_input('Tenure (months)', min_value=0, step=1)
    MonthlyCharges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, step=0.1)
    TotalCharges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, step=0.1)
    
    # Categorical fields
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    Partner = st.sidebar.selectbox('Has Partner?', ['No', 'Yes'])
    Dependents = st.sidebar.selectbox('Has Dependents?', ['No', 'Yes'])
    PhoneService = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

    # One-hot encoding format
    d_dict = {'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges]}
    
    # One-hot encoding categorical variables
    categories = {"gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
                  "Dependents": Dependents, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
                  "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
                  "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
                  "TechSupport": TechSupport, "StreamingTV": StreamingTV,
                  "StreamingMovies": StreamingMovies, "Contract": Contract,
                  "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod}
    
    for key, value in categories.items():
        for option in st.sidebar.selectbox.options:
            d_dict[f'{key}_{option}'] = [1 if value == option else 0]
    
    return pd.DataFrame.from_dict(d_dict)

def predict_churn(df):
    """Predict customer churn based on the input features."""
    churned = model.predict(df)[0]
    prediction_prob = model.predict_proba(df)[:, 1][0]
    risk_level = "Very Low"
    
    if prediction_prob >= 85:
        risk_level = "Very High"
    elif prediction_prob >= 75:
        risk_level = "High"
    elif prediction_prob >= 50:
        risk_level = "Moderate"
    elif prediction_prob >= 25:
        risk_level = "Low"
    
    return churned, round(prediction_prob * 100, 2), risk_level

def main():
    """Streamlit UI setup."""
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to predict the likelihood of churn.")
    
    df = get_data()
    
    if st.sidebar.button("Predict Churn"):
        churned, prediction_prob, risk_level = predict_churn(df)
        
        st.subheader("Prediction Result")
        st.write("This customer is **likely to churn**" if churned else "This customer is **not likely to churn**")
        st.write(f"**Churn Probability:** {prediction_prob}%")
        st.write(f"**Risk Level:** {risk_level}")
        
        st.subheader("Customer Data Preview")
        st.dataframe(df)
    
if __name__ == "__main__":
    main()
