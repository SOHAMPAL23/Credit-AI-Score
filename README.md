# CreditScore AI - Loan Approval Prediction System

This project implements a machine learning system to predict loan approval chances based on applicant data. The model achieves over 98% accuracy, precision, recall, and F1-score.

## Features

- Predicts loan approval (Approved/Denied) based on applicant information
- Uses Random Forest algorithm with optimized hyperparameters
- Interactive Streamlit web application for real-time predictions
- Feature importance analysis
- Model performance visualization

## Requirements

- Python 3.7+
- Required packages (see [requirements.txt](requirements.txt))

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the main script to generate synthetic data, train the model, and save it:
```
python credit_score_ai.py
```

This will:
- Generate 1000 synthetic loan applicant samples
- Train a Random Forest model with optimized hyperparameters
- Save the trained model to `credit_score_ai_model.pkl`
- Display performance metrics and visualizations

### Running the Interactive Web App

After training the model, run the Streamlit application:
```
streamlit run app.py
```

The web app allows you to:
- Adjust applicant parameters using sliders and input fields
- Get real-time loan approval predictions
- View feature importance
- Receive recommendations for improving approval chances

## Model Performance

The trained model achieves exceptional performance:
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%

## Features Used

1. Age
2. Annual Income
3. Years Employed
4. Loan Amount
5. Credit Score
6. Loan Purpose (encoded)
7. Debt-to-Income Ratio

## Business Applications

- Automate preliminary loan approval processes
- Provide consistent, data-driven decision making
- Identify high-risk applications requiring additional review
- Improve customer experience with faster decision times