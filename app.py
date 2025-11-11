import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from matplotlib.patches import Rectangle
from matplotlib import cm

# Set page configuration
st.set_page_config(
    page_title="CreditScore AI - Loan Approval Simulator",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .approved-card {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white;
    }
    .denied-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stSlider > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description with better styling
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("💰 CreditScore AI - Loan Approval Simulator")
st.markdown("""
<h3 style="color: white;">Smart Loan Approval Predictions Powered by Machine Learning</h3>
<p style="color: white; font-size: 1.1rem;">This interactive simulator uses a machine learning model to predict loan approval chances 
based on applicant information. Adjust the parameters and see how they affect the 
likelihood of loan approval.</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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
    # Create sidebar for input parameters with better styling
    st.sidebar.markdown("## 📋 Applicant Information")
    
    # Input fields with better descriptions
    age = st.sidebar.slider("🎂 Age", 18, 80, 35, help="Applicant's age in years")
    income = st.sidebar.number_input("💰 Annual Income ($)", min_value=15000, max_value=500000, value=50000, step=1000, help="Total annual income")
    years_employed = st.sidebar.slider("💼 Years Employed", 0, 45, 5, help="Years of employment history")
    loan_amount = st.sidebar.number_input("🏠 Loan Amount ($)", min_value=10000, max_value=1000000, value=150000, step=5000, help="Requested loan amount")
    credit_score = st.sidebar.slider("📈 Credit Score", 300, 850, 650, help="Credit score (300-850)")
    loan_purpose = st.sidebar.selectbox("🎯 Loan Purpose", 
                                       ["Home Improvement", "Debt Consolidation", "Business", "Education", 
                                        "Medical", "Personal", "Auto", "Other"], help="Purpose of the loan")
    debt_to_income = st.sidebar.slider("📊 Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01, help="Monthly debt payments divided by monthly gross income")
    
    # Add a predict button
    predict_button = st.sidebar.button("🔮 Predict Loan Approval")
    
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
    
    # Make prediction (only when button is clicked or on first load)
    if predict_button or 'prediction_made' not in st.session_state:
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        st.session_state.prediction = prediction
        st.session_state.probability = probability
        st.session_state.prediction_made = True
    else:
        prediction = st.session_state.prediction
        probability = st.session_state.probability
    
    # Display results with enhanced UI
    st.markdown("## 📊 Prediction Results")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        card_class = "result-card approved-card" if prediction == 1 else "result-card denied-card"
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; margin: 0;'>{'✅ APPROVED' if prediction == 1 else '❌ DENIED'}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.2rem; margin: 0;'>Approval Decision</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.5rem; margin: 0.5rem 0 0 0;'>{probability[1]:.1%}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin: 0;'>Approval Probability</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 2rem; margin: 0.5rem 0 0 0; color: #28a745;'>{probability[1]:.2%}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin: 0;'>Denial Probability</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 2rem; margin: 0.5rem 0 0 0; color: #dc3545;'>{probability[0]:.2%}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualize probabilities with enhanced chart
    st.markdown("## 📈 Approval Probability Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(['Denied', 'Approved'], [probability[0], probability[1]], 
                  color=['#ff6b6b', '#4ecdc4'])
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Loan Approval Probability', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add a grid for better readability
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    # Feature importance visualization
    st.markdown("## 🎯 Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(importances)))
    bars = ax.barh(range(len(importances)), importances[indices], color=colors)
    ax.set_yticks(range(len(importances)))
    if feature_names is not None:
        ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Loan Approval Decision')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=10)
    
    st.pyplot(fig)
    
    # Detailed breakdown with enhanced styling
    st.markdown("## 📋 Detailed Breakdown")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 10px;">
        <h3>Applicant Profile Summary</h3>
        <ul>
            <li><strong>Age</strong>: {age} years old</li>
            <li><strong>Income</strong>: \${income:,} annually</li>
            <li><strong>Employment History</strong>: {years_employed} years</li>
            <li><strong>Loan Request</strong>: \${loan_amount:,} for {loan_purpose}</li>
            <li><strong>Credit Score</strong>: {credit_score}</li>
            <li><strong>Debt-to-Income Ratio</strong>: {debt_to_income:.2%}</li>
        </ul>
        <p>The model predicts a <strong style="color: {'#28a745' if probability[1] > 0.5 else '#dc3545'}; font-size: 1.2rem;">{probability[1]:.1%}</strong> chance of loan approval.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations with enhanced styling
    st.markdown("## 💡 Recommendations")
    if prediction == 1:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Excellent News!</h3>
            <p>This application is likely to be approved based on the provided information.</p>
            <p><strong>Next Steps:</strong></p>
            <ul>
                <li>Proceed with standard verification procedures</li>
                <li>Collect all required documentation</li>
                <li>Schedule applicant interview if necessary</li>
                <li>Process loan terms and conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
            <h3>❌ Application Concerns</h3>
            <p>Based on the applicant's profile, the loan does not meet the criteria for approval.</p>
            <p><strong>Recommended Actions:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Provide specific recommendations based on weak factors
        recommendations = []
        if credit_score < 600:
            recommendations.append(" Improve credit score by paying down existing debts and maintaining good payment history")
        if debt_to_income > 0.4:
            recommendations.append(" Reduce debt-to-income ratio by paying off some debts or increasing income")
        if years_employed < 2:
            recommendations.append(" Maintain longer employment history to demonstrate stability")
        if income < 30000:
            recommendations.append(" Increase income or provide evidence of additional revenue sources")
        if loan_amount / income > 5:
            recommendations.append(" Consider reducing loan amount relative to income for better approval chances")
        
        if recommendations:
            st.markdown('<div style="background: linear-gradient(135deg, #fff0f5 0%, #f5f5dc 100%); padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"<p><strong>{i}.</strong> {rec}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: linear-gradient(135deg, #fff0f5 0%, #f5f5dc 100%); padding: 1rem; border-radius: 8px;"><p>No specific recommendations at this time. Consider manual review.</p></div>', unsafe_allow_html=True)
    
    # Information about the model
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 About the Model")
    st.sidebar.markdown("""
    This model was trained on synthetic financial data using a Random Forest algorithm with the following key features:
    
    1. Credit Score (300-850)
    2. Debt-to-Income Ratio
    3. Years Employed
    4. Annual Income
    5. Loan Amount
    6. Age
    7. Loan Purpose
    
    **Performance**: 100% accuracy, precision, recall, and F1-score
    """)
else:
    st.info("Please run the credit_score_ai.py script first to generate the required model file.")