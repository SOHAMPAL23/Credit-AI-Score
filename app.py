import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="CreditScore AI - Loan Approval Simulator",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 CreditScore AI - Loan Approval Simulator")
st.markdown("""
This interactive simulator uses a machine learning model to predict loan approval chances 
based on applicant information. Adjust the parameters and see how they affect the 
likelihood of loan approval.
""")

# Initialize variables
model = None
scaler = None
label_encoder = None
feature_names = None

# Try to load the trained model
if os.path.exists('credit_score_ai_model.pkl'):
    try:
        model_data = joblib.load('credit_score_ai_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False
else:
    st.warning("Trained model not found. Please run the credit_score_ai.py script first to generate the model.")
    model_loaded = False

if model_loaded and model is not None and scaler is not None and label_encoder is not None:
    # Create sidebar for input parameters
    st.sidebar.header("Applicant Information")

    # Input fields
    age = st.sidebar.slider("Age", 18, 80, 35)
    income = st.sidebar.number_input("Annual Income ($)", min_value=15000, max_value=500000, value=50000, step=1000)
    years_employed = st.sidebar.slider("Years Employed", 0, 45, 5)
    loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=10000, max_value=1000000, value=150000, step=5000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    loan_purpose = st.sidebar.selectbox("Loan Purpose", 
                                       ["Home Improvement", "Debt Consolidation", "Business", "Education", 
                                        "Medical", "Personal", "Auto", "Other"])
    debt_to_income = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01)

    # Encode loan purpose
    try:
        loan_purpose_encoded = label_encoder.transform([loan_purpose])[0]
    except ValueError:
        loan_purpose_encoded = 0  # Default encoding

    # Create feature array
    features = np.array([[age, income, years_employed, loan_amount, 
                         credit_score, loan_purpose_encoded, debt_to_income]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    # Display results
    st.header("Prediction Results")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Approval Decision", "APPROVED" if prediction == 1 else "DENIED", 
                  delta=f"{probability[1]:.1%}" if prediction == 1 else f"-{probability[0]:.1%}")

    with col2:
        st.metric("Approval Probability", f"{probability[1]:.2%}")

    with col3:
        st.metric("Denial Probability", f"{probability[0]:.2%}")

    # Visualize probabilities
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(['Denied', 'Approved'], [probability[0], probability[1]], 
                  color=['#ff6b6b', '#4ecdc4'])
    ax.set_ylabel('Probability')
    ax.set_title('Loan Approval Probability')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')

    st.pyplot(fig)

    # Feature importance visualization
    st.header("Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color='skyblue')
    ax.set_yticks(range(len(importances)))
    if feature_names is not None:
        ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Loan Approval Decision')
    plt.gca().invert_yaxis()  # Highest importance at top

    st.pyplot(fig)

    # Detailed breakdown
    st.header("Detailed Breakdown")
    st.markdown(f"""
    Based on the provided information:
    - **Age**: {age} years old
    - **Income**: \${income:,} annually
    - **Employment History**: {years_employed} years
    - **Loan Request**: \${loan_amount:,} for {loan_purpose}
    - **Credit Score**: {credit_score}
    - **Debt-to-Income Ratio**: {debt_to_income:.2%}

    The model predicts a **{probability[1]:.1%}** chance of loan approval.
    """)

    # Recommendations
    st.header("Recommendations")
    if prediction == 1:
        st.success("✅ This application is likely to be approved!")
        st.markdown("""
        Based on the applicant's profile, the loan meets the criteria for approval. 
        However, always conduct standard verification procedures before final approval.
        """)
    else:
        st.error("❌ This application is likely to be denied.")
        st.markdown("""
        Based on the applicant's profile, the loan does not meet the criteria for approval. 
        Consider the following recommendations to improve approval chances:
        """)
        
        # Provide specific recommendations based on weak factors
        recommendations = []
        if credit_score < 600:
            recommendations.append("• Improve credit score by paying down existing debts")
        if debt_to_income > 0.4:
            recommendations.append("• Reduce debt-to-income ratio by paying off some debts")
        if years_employed < 2:
            recommendations.append("• Maintain longer employment history")
        if income < 30000:
            recommendations.append("• Increase income or provide evidence of additional revenue sources")
        
        for rec in recommendations:
            st.markdown(rec)

    # Information about the model
    st.sidebar.markdown("---")
    st.sidebar.header("About the Model")
    st.sidebar.markdown("""
    This model was trained on synthetic financial data using a Random Forest algorithm.
    Key features considered in the decision:
    1. Credit Score (300-850)
    2. Debt-to-Income Ratio
    3. Years Employed
    4. Annual Income
    5. Loan Amount
    6. Age
    7. Loan Purpose
    """)
else:
    st.info("Please run the credit_score_ai.py script first to generate the required model file.")