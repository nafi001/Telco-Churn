import streamlit as st
import pandas as pd
import joblib
import os
import base64
from PIL import Image
import numpy as np

# Set page configuration - MUST COME FIRST
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_smote_model.joblib")

model = load_model()

# Add custom CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        .big-font {{
            font-size:26px !important;
            font-weight: bold;
        }}
        .result-box {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .green-box {{
            background-color: rgba(0, 255, 0, 0.2);
        }}
        .red-box {{
            background-color: rgba(255, 0, 0, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# If you have a background image file, uncomment this
# add_bg_from_local('background.png')  

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Churn", "About"])

if page == "Home":
    st.title("Customer Churn Prediction")
    st.write("Welcome to the Customer Churn Prediction App!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Customer Churn?
        
        Customer churn refers to when customers stop doing business with a company. 
        Predicting which customers are likely to churn helps businesses take proactive 
        steps to retain valuable customers.
        
        ### How to use this app?
        
        1. Navigate to the "Predict Churn" page using the sidebar
        2. Enter customer information in the form
        3. Click on "Predict Churn" to see results
        
        ### Model Information
        
        This app uses a Random Forest classifier trained with SMOTEENN resampling technique 
        to predict customer churn based on telecommunication customer data.
        """)
    
    with col2:
        st.markdown("""
        ### Key Features
        
        - Easy-to-use interface
        - Real-time predictions
        - Churn probability estimates
        - Risk assessment
        """)

elif page == "Predict Churn":
    st.title("Customer Churn Prediction")
    st.write("Enter customer information to predict churn probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["0", "1"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        
    with col3:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=29.85)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=29.85)
    
    # Create dictionary for all features
    def create_feature_dict():
        # Initialize with numerical features
        d_dict = {
            'tenure': [tenure], 
            'MonthlyCharges': [monthly_charges], 
            'TotalCharges': [total_charges]
        }
        
        # Create all the one-hot encoded columns with zeros
        features = [
            f'gender_{gender}',
            f'SeniorCitizen_{senior_citizen}',
            f'Partner_{partner}',
            f'Dependents_{dependents}',
            f'PhoneService_{phone_service}',
            f'MultipleLines_{multiple_lines}',
            f'InternetService_{internet_service}',
            f'OnlineSecurity_{online_security}',
            f'OnlineBackup_{online_backup}',
            f'DeviceProtection_{device_protection}',
            f'TechSupport_{tech_support}',
            f'StreamingTV_{streaming_tv}',
            f'StreamingMovies_{streaming_movies}',
            f'Contract_{contract}',
            f'PaperlessBilling_{paperless_billing}',
            f'PaymentMethod_{payment_method}'
        ]
        
        # All possible categorical values
        all_features = ['gender_Female', 'gender_Male', 'SeniorCitizen_0', 'SeniorCitizen_1', 'Partner_No',
                      'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
                      'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service',
                      'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
                      'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service',
                      'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
                      'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service',
                      'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service',
                      'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service',
                      'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service',
                      'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
                      'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
                      'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                      'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
        
        # Initialize all categorical features with 0
        for feature in all_features:
            d_dict[feature] = [0]
        
        # Set the selected features to 1
        for feature in features:
            if feature in d_dict:
                d_dict[feature] = [1]
                
        return pd.DataFrame.from_dict(d_dict)
    
    if st.button("Predict Churn"):
        # Get the feature dataframe
        input_df = create_feature_dict()
        
        # Display the input data
        with st.expander("View Input Data"):
            st.dataframe(input_df)
            
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
        pred_percent = round(prediction_proba * 100, 2)
        
        # Determine risk level
        if pred_percent >= 85:
            risk = "Very High"
            risk_color = "#FF0000"  # Red
        elif pred_percent >= 75:
            risk = "High"
            risk_color = "#FF4500"  # OrangeRed
        elif pred_percent >= 50:
            risk = "Moderate"
            risk_color = "#FFA500"  # Orange
        elif pred_percent >= 25:
            risk = "Low"
            risk_color = "#4682B4"  # SteelBlue
        else:
            risk = "Very Low"
            risk_color = "#008000"  # Green
        
        # Show results
        st.markdown("### Prediction Results")
        
        # Columns for displaying results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box red-box">
                    <p class="big-font">This customer is likely to churn</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box green-box">
                    <p class="big-font">This customer is not likely to churn</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-box" style="background-color: rgba(70, 130, 180, 0.2);">
                <p class="big-font">Churn Probability: {pred_percent}%</p>
                <p style="font-size:20px; color:{risk_color}; font-weight:bold;">Risk Level: {risk}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance plot
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            
            feature_importance = pd.DataFrame({
                'feature': input_df.columns,
                'importance': model.feature_importances_
            })
            
            # Sort features by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
            
            # Plot feature importance
            st.bar_chart(feature_importance.set_index('feature'))
            
            # Key influential factors
            st.markdown("### Key Influential Factors")
            top_features = feature_importance.head(5)['feature'].tolist()
            
            for i, feature in enumerate(top_features):
                if feature in input_df.columns and 'tenure' in feature:
                    st.write(f"{i+1}. **{feature}**: {tenure} months")
                elif feature in input_df.columns and 'MonthlyCharges' in feature:
                    st.write(f"{i+1}. **{feature}**: ${monthly_charges}")
                elif feature in input_df.columns and 'TotalCharges' in feature:
                    st.write(f"{i+1}. **{feature}**: ${total_charges}")
                elif input_df[feature].values[0] == 1:
                    st.write(f"{i+1}. **{feature}**: Yes")

elif page == "About":
    st.title("About")
    st.write("This is a Streamlit app for predicting customer churn.")
    
    st.markdown("""
    ### Model Information
    
    This application uses a Random Forest model that was trained on telecommunications customer data.
    The model was enhanced using SMOTEENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors) 
    to address class imbalance in the dataset.
    
    ### Data Features
    
    The model takes into account the following customer attributes:
    
    - **Demographic Information**: Gender, Senior Citizen status, Partner, Dependents
    - **Account Information**: Tenure, Contract type, Paperless Billing, Payment Method
    - **Services**: Phone Service, Multiple Lines, Internet Service, etc.
    - **Charges**: Monthly Charges, Total Charges
    
    ### How to Deploy
    
    To deploy this app on GitHub:
    
    1. Create a GitHub repository
    2. Upload this code as `app.py`
    3. Include your model file `random_forest_smote_model.joblib`
    4. Create a `requirements.txt` file with the necessary dependencies
    5. Set up GitHub Actions for CI/CD or connect with a deployment platform like Streamlit Cloud
    
    ### Credits
    
    Created by: [Your Name]
    """)
    
    st.markdown("""
    ### Requirements
    
    ```
    streamlit
    pandas
    scikit-learn
    joblib
    numpy
    pillow
    ```
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; font-size: 12px;">
    © 2025 Customer Churn Prediction App. All rights reserved.
</div>
""", unsafe_allow_html=True)
