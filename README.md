# Customer Churn Analysis in the Telecom Industry

This repository contains an exploratory analysis of customer churn for a telecom company. The goal was to understand customer behavior, identify factors contributing to churn, and derive actionable insights to improve retention strategies.
a
---

## Table of Contents
- [Overview](#overview)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Understanding Tenure and Billing](#understanding-tenure-and-billing)
  - [Feature Transformations](#feature-transformations)
- [Key Insights](#key-insights)
- [Business Recommendations](#business-recommendations)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Algorithms Used](#algorithms-used)
  - [Model Performance Before Oversampling](#model-performance-before-oversampling)
  - [SMOTEENN for Imbalanced Data](#smoteenn-for-imbalanced-data)
  - [Model Performance After Oversampling](#model-performance-after-oversampling)
- [Final Model Selection](#final-model-selection)


---

## Overview

This project analyzes a dataset from a telecom company to predict customer churn. By transforming raw data, identifying missing values, and performing thorough EDA, we pinpoint the critical factors that affect churn. Our findings support strategies to improve customer retention and optimize business decisions.

---

## Data Overview

The dataset provides customer details and service usage information, including:
- **Customer Demographics:** Gender, Age, Senior Citizen status, Marital status, and Dependents.
- **Service Usage:** Phone service, Internet service (DSL or Fiber Optic), and additional services (Online Security, Backup, Tech Support, etc.).
- **Billing Information:** Contract type, Payment method, Monthly charges, and Total charges.
- **Customer Relationship:** Tenure and churn status.

---

## Data Preprocessing

1. **Initial Data Inspection:**  
   - The data appeared to have no missing values. However, most columns were stored as objects.

2. **Conversion of Data Types:**  
   - Object types were converted to numeric using `to_numeric` to detect hidden missing values.
   - **Result:** Found 11 missing values in the `TotalCharges` column.

3. **Handling Missing Values:**  
   - Investigation revealed that missing `TotalCharges` correspond to customers with **tenure = 0** (new customers who have not yet paid).
   - These missing records represent only ~0.156% of the data and were dropped.

4. **Feature Engineering:**  
   - **Tenure Transformation:**  
     - Raw tenure values were binned into groups (e.g., 0–12 months as “0–1 year”, 13–24 months as “1–2 years”, etc.) for easier visualization and analysis.
   - **Column Removal:**  
     - Dropped `CustomerID` (irrelevant for analysis) and the original `tenure` column (after creating `tenure_group`).

---

## Exploratory Data Analysis (EDA)

### Understanding Tenure and Billing

- **Customer Tenure:**  
  - 75% of customers have a tenure of less than 55 months.
  - The mean tenure is approximately **32 months**.
  
- **Billing Information:**  
  - The average monthly charge is about **$64.76**.
  - 25% of customers pay **more than $89.85** per month.
  
- **Dataset Imbalance:**  
  - The data distribution is imbalanced, which is an important factor to consider during model training.

### Feature Transformations

- **Tenure Binning:**  
  - To better visualize and understand customer longevity, tenure was binned into discrete groups.
  - **Observation:** Customers with less than a year of tenure are more likely to churn, while long-term customers rarely churn.

- **Data Type Conversion:**  
  - Converting columns from object to numeric revealed hidden missing values and allowed more precise analysis.

---

## Key Insights

1. **Demographic and Behavioral Trends:**
   - **Gender:** No significant impact alone, but potentially relevant when combined with other features.
   - **Senior Citizens:** More likely to churn, even though they comprise only 16% of the customer base.
   - **Family Status:** Customers with partners are more likely to churn, while those with dependents churn less.

2. **Service and Usage Patterns:**
   - **Internet Service:**  
     - Customers using Fiber Optic services show a higher churn rate—likely due to cost.
     - DSL customers paying higher charges tend to be more loyal.
   - **Additional Services:**  
     - Subscriptions to Online Security, Online Backup, and Tech Support are associated with lower churn rates.
     - Streaming TV/Movies do not significantly impact churn.

3. **Contract and Payment Method Impact:**
   - **Contract Type:**  
     - Month-to-month contracts have the highest churn risk, increasing the likelihood by 6.31x compared to long-term contracts.
     - Customers on two-year contracts show very low churn rates.
   - **Payment Methods:**  
     - Customers paying by electronic check are more likely to churn.
     - Credit card users have a lower churn rate.

4. **Price Sensitivity:**
   - Customers with lower monthly charges tend to churn, possibly due to the trial nature of their subscription.
   - There is a clear relationship between monthly charges and total charges: as monthly charges increase, total charges increase proportionally.

---

## Business Recommendations

Based on our analysis, we propose the following strategies:
- **Target New Customers:**  
  - Engage new customers (tenure < 1 year) with onboarding offers to reduce early churn.
- **Incentivize Long-Term Contracts:**  
  - Encourage customers on month-to-month contracts to switch to longer-term agreements.
- **Focus on Payment Methods:**  
  - Provide incentives for customers to move from electronic checks to credit cards.
- **Enhance Service Packages:**  
  - Bundle additional services like Online Security, Backup, and Tech Support to enhance customer loyalty.
- **Tailor Strategies for Senior Citizens:**  
  - Implement personalized retention programs for senior customers.


---


## Handling Imbalanced Data

The dataset is imbalanced, with far fewer churners than non-churners. This affects model performance, particularly for the minority class (churners). To address this, we apply **SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors)** to balance the class distribution.

---

## Model Training and Evaluation

### Algorithms Used

We experimented with the following models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)
- AdaBoost

### Model Performance Before Oversampling

- **Issue:**  
  - The models showed high accuracy but poor performance in detecting churners.
  - The F1-score for the minority class (`Churn = 1`) was low (~0.58).
  - This happens because most machine learning models tend to ignore the minority class in imbalanced datasets.

---

## SMOTEENN for Imbalanced Data

To improve the model, we applied **SMOTEENN**, which:

- **Oversamples the minority class (churners) using SMOTE**: This creates synthetic examples to balance the class distribution.
- **Cleans noisy samples using Edited Nearest Neighbors (ENN)**: This removes mislabeled data points, improving model robustness.

**After applying SMOTEENN, the dataset became nearly balanced, significantly improving model performance.**

---

## Model Performance After Oversampling

- **F1-score for churners increased significantly** after balancing the dataset.
- **Decision Tree** emerged as one of the best-performing models, showing high accuracy and improved precision/recall for churners.
- PCA helped in feature reduction but did not significantly outperform other models.

---


## Final Model Selection

**Based on performance metrics, we selected Decision Tree as our final model** due to its superior ability to handle imbalanced data and its high predictive accuracy.
