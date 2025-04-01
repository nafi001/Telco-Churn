import streamlit as st
import pandas as pd
import joblib
import os
import base64
import numpy as np
a
# Set page configuration - MUST COME FIRST
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # Make sidebar visible by default
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_smote_model.joblib")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1E3A8A;
        text-align: center;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #1E3A8A;
    }
    .sidebar-content {
        font-size: 18px;
        font-weight: bold;
    }
    .stRadio > div {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e0f7fa;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .green-box {
        background-color: rgba(0, 200, 0, 0.2);
    }
    .red-box {
        background-color: rgba(255, 0, 0, 0.2);
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 15px;
    }
    .stButton>button:hover {
        background-color: #2E4A9A;
    }
    /* Improve form field visibility */
    .stSelectbox, .stNumberInput {
        margin-bottom: 15px;
    }
    /* Make sidebar more visible */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding-top: 20px;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Make sidebar more prominent
st.sidebar.markdown('<p class="sidebar-content">NAVIGATION MENU</p>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["🏠 Predict Churn", "ℹ️ About"], index=0)

# Load model
model = load_model()

# Main content
if page == "🏠 Predict Churn":
    st.markdown('<p class="main-header">Customer Churn Prediction</p>', unsafe_allow_html=True)
    
    # Quick explanation in a collapsible section
    with st.expander("What is Customer Churn?", expanded=False):
        st.markdown("""
        Customer churn refers to when customers stop doing business with a company. 
        Predicting which customers are likely to churn helps businesses take proactive 
        steps to retain valuable customers.
        
        This app uses a Random Forest classifier to predict customer churn based on 
        telecommunications customer data.
        """)
    
    # Form for prediction
    st.markdown('<p class="sub-header">Customer Information</p>', unsafe_allow_html=True)
    st.markdown('<div class="highlight-box">Enter the customer details below and click "Predict Churn" to see results.</div>', unsafe_allow_html=True)
    
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
    
    # Create dataframe with get_dummies for proper one-hot encoding
    def prepare_input_data():
        # Create base dataframe with all customer data
        input_data = pd.DataFrame({
            'customerID': ['TEMP001'],
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Replace Churn values exactly as done during model training
        replaceStruct = {"Churn": {"No": 0, "Yes": 1}}
        
        # Define columns for one-hot encoding (same as in your training code)
        oneHotCols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                      "InternetService", "OnlineSecurity", "OnlineBackup", 
                      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                      "Contract", "PaperlessBilling", "PaymentMethod"]
        
        # Apply get_dummies to create the same encoding as your model was trained with
        encoded_df = pd.get_dummies(input_data, columns=oneHotCols)
        
        # Drop the customerID column which is not needed for prediction
        encoded_df = encoded_df.drop('customerID', axis=1)
        
        return encoded_df
    
    # Ensure all columns needed by model are present
    def ensure_all_columns(df, model_columns):
        # Find missing columns
        missing_cols = set(model_columns) - set(df.columns)
        
        # Add missing columns with default value of 0
        for col in missing_cols:
            df[col] = 0
            
        # Ensure columns are in the same order as the model expects
        return df[model_columns]
    
    # Big button for prediction
    st.markdown('<br>', unsafe_allow_html=True)
    predict_button = st.button("Predict Churn", key="predict_button")
    
    if predict_button:
        # Get the input data in dataframe form
        input_df = prepare_input_data()
        
        # We need to make sure we have all columns that the model expects
        # For this, we can extract column names from the model or define them based on your training code
        try:
            # Try to get feature names from the model
            model_columns = model.feature_names_in_
        except AttributeError:
            # If feature names aren't available, we'll show a message and proceed with what we have
            st.warning("Could not determine exact model columns. Prediction may be inaccurate.")
            model_columns = input_df.columns
        
        # Ensure all required columns are present with proper ordering
        final_df = ensure_all_columns(input_df, model_columns) if hasattr(model, 'feature_names_in_') else input_df
        
        # Display the input data
        with st.expander("View Processed Input Data"):
            st.dataframe(final_df)
            
        # Make prediction
        try:
            prediction = model.predict(final_df)[0]
            prediction_proba = model.predict_proba(final_df)[0][1]
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
            st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
            
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
                st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
                
                feature_importance = pd.DataFrame({
                    'feature': final_df.columns,
                    'importance': model.feature_importances_
                })
                
                # Sort features by importance
                feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
                
                # Plot feature importance
                st.bar_chart(feature_importance.set_index('feature'))
                
                # Key influential factors
                st.markdown('<p class="sub-header">Key Influential Factors</p>', unsafe_allow_html=True)
                top_features = feature_importance.head(5)['feature'].tolist()
                
                for i, feature in enumerate(top_features):
                    if feature in final_df.columns and 'tenure' in feature:
                        st.write(f"{i+1}. **{feature}**: {tenure} months")
                    elif feature in final_df.columns and 'MonthlyCharges' in feature:
                        st.write(f"{i+1}. **{feature}**: ${monthly_charges}")
                    elif feature in final_df.columns and 'TotalCharges' in feature:
                        st.write(f"{i+1}. **{feature}**: ${total_charges}")
                    elif feature.startswith(tuple(oneHotCols)) and feature in final_df.columns and final_df[feature].values[0] == 1:
                        feature_name = feature.split('_')[0]
                        feature_value = '_'.join(feature.split('_')[1:])
                        st.write(f"{i+1}. **{feature_name}**: {feature_value}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Make sure the model file matches the one you used during training.")

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About This App</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Customer Churn Prediction Model
    
    This application uses a Random Forest model that was trained on telecommunications customer data.
    The model was enhanced using SMOTEENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors) 
    to address class imbalance in the dataset.
    
    ### Data Features
    
    The model analyzes the following customer attributes:
    
    - **Demographic Information**: Gender, Senior Citizen status, Partner, Dependents
    - **Account Information**: Tenure, Contract type, Paperless Billing, Payment Method
    - **Services**: Phone Service, Multiple Lines, Internet Service, etc.
    - **Charges**: Monthly Charges, Total Charges
    
    ### Deployment Instructions
    
    To deploy this app on GitHub:
    
    1. Create a GitHub repository
    2. Upload this code as `app.py`
    3. Include your model file `random_forest_smote_model.joblib`
    4. Create a `requirements.txt` file with the necessary dependencies
    5. Connect with Streamlit Cloud for easy deployment
    """)
    
    st.markdown("""
    ### Requirements
    
    ```
    streamlit==1.27.0
    pandas==1.5.3
    scikit-learn==1.2.2
    joblib==1.3.1
    numpy==1.24.3
    ```
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; font-size: 12px; color: #666;">
    © 2025 Customer Churn Prediction App. All rights reserved.
</div>
""", unsafe_allow_html=True)
