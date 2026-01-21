"""
Loan Approval Prediction - Deployment Script
This script demonstrates how to load and use the optimized loan approval model.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any

class LoanApprovalPredictor:
    """Enhanced loan approval prediction system"""
    
    def __init__(self, model_path: str = 'loan_approval_final_model.pkl'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model package
        """
        try:
            self.model_package = joblib.load(model_path)
            self.model = self.model_package['model']
            self.scaler = self.model_package['scaler']
            self.label_encoder = self.model_package['label_encoder']
            self.feature_names = self.model_package['feature_names']
            self.model_name = self.model_package['model_name']
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model Type: {self.model_name}")
            print(f"📈 Features: {len(self.feature_names)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _engineer_features(self, age: float, income: float, years_employed: float, 
                          loan_amount: float, credit_score: float, loan_purpose: str, 
                          debt_to_income: float) -> np.ndarray:
        """
        Engineer features from raw input data
        """
        # Encode loan purpose
        try:
            loan_purpose_encoded = self.label_encoder.transform([loan_purpose])[0]
        except ValueError:
            # Handle unseen categories
            loan_purpose_encoded = 0  # Default encoding
        
        # Calculate engineered features
        income_to_loan_ratio = income / loan_amount if loan_amount > 0 else 0
        credit_utilization = (debt_to_income * income) / credit_score if credit_score > 0 else 0
        employment_stability = years_employed / max(age - 18, 1)
        
        # Simplified categorical encodings
        age_group_encoded = 2 if 25 <= age <= 50 else (1 if age < 25 else 3)
        income_bracket_encoded = 2 if income > 80000 else (1 if income > 40000 else 0)
        credit_category_encoded = 3 if credit_score > 740 else (2 if credit_score > 670 else 1)
        dti_risk_encoded = 2 if debt_to_income > 0.43 else (1 if debt_to_income > 0.36 else 0)
        
        # Return feature array in correct order
        features = np.array([[
            age, income, years_employed, loan_amount, credit_score,
            loan_purpose_encoded, debt_to_income, income_to_loan_ratio,
            credit_utilization, employment_stability, age_group_encoded,
            income_bracket_encoded, credit_category_encoded, dti_risk_encoded
        ]])
        
        return features
    
    def predict(self, age: float, income: float, years_employed: float, 
                loan_amount: float, credit_score: float, loan_purpose: str, 
                debt_to_income: float) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """
        Make loan approval prediction
        
        Returns:
            Tuple of (prediction, probabilities, detailed_info)
        """
        # Validate inputs
        self._validate_inputs(age, income, years_employed, loan_amount, credit_score, debt_to_income)
        
        # Engineer features
        features = self._engineer_features(
            age, income, years_employed, loan_amount, credit_score, loan_purpose, debt_to_income
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create detailed information
        detailed_info = {
            'model_used': self.model_name,
            'confidence': max(probabilities),
            'risk_level': self._assess_risk(probabilities[1]),
            'recommendations': self._generate_recommendations(
                age, income, years_employed, loan_amount, credit_score, debt_to_income
            )
        }
        
        return int(prediction), probabilities, detailed_info
    
    def _validate_inputs(self, age: float, income: float, years_employed: float,
                        loan_amount: float, credit_score: float, debt_to_income: float):
        """Validate input parameters"""
        if not (18 <= age <= 100):
            raise ValueError("Age must be between 18 and 100")
        if income <= 0:
            raise ValueError("Income must be positive")
        if years_employed < 0:
            raise ValueError("Years employed cannot be negative")
        if loan_amount <= 0:
            raise ValueError("Loan amount must be positive")
        if not (300 <= credit_score <= 850):
            raise ValueError("Credit score must be between 300 and 850")
        if not (0 <= debt_to_income <= 1):
            raise ValueError("Debt-to-income ratio must be between 0 and 1")
    
    def _assess_risk(self, approval_probability: float) -> str:
        """Assess risk level based on approval probability"""
        if approval_probability >= 0.8:
            return "Low Risk"
        elif approval_probability >= 0.6:
            return "Medium Risk"
        elif approval_probability >= 0.4:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _generate_recommendations(self, age: float, income: float, years_employed: float,
                                 loan_amount: float, credit_score: float, debt_to_income: float) -> list:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Income to loan ratio check
        income_ratio = income / loan_amount
        if income_ratio < 2:
            recommendations.append("Increase income or reduce loan amount to improve income-to-loan ratio")
        
        # Credit score improvement
        if credit_score < 650:
            recommendations.append("Work on improving credit score (aim for 650+)")
        elif credit_score < 700:
            recommendations.append("Consider credit repair services to boost score above 700")
        
        # DTI ratio optimization
        if debt_to_income > 0.4:
            recommendations.append("Pay down existing debts to reduce debt-to-income ratio below 40%")
        
        # Employment stability
        if years_employed < 2:
            recommendations.append("Longer employment history improves approval chances")
        
        # Return recommendations or default message
        if not recommendations:
            recommendations.append("Strong application with good approval prospects")
        
        return recommendations

def demo_predictions():
    """Demonstrate the predictor with sample cases"""
    print("🏦 LOAN APPROVAL PREDICTOR DEMO")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = LoanApprovalPredictor()
        
        # Test cases
        test_cases = [
            {
                'name': 'Strong Candidate',
                'age': 35,
                'income': 90000,
                'years_employed': 10,
                'loan_amount': 150000,
                'credit_score': 760,
                'loan_purpose': 'Home Improvement',
                'debt_to_income': 0.15
            },
            {
                'name': 'Borderline Case',
                'age': 28,
                'income': 50000,
                'years_employed': 3,
                'loan_amount': 120000,
                'credit_score': 680,
                'loan_purpose': 'Debt Consolidation',
                'debt_to_income': 0.35
            },
            {
                'name': 'Weak Candidate',
                'age': 22,
                'income': 25000,
                'years_employed': 1,
                'loan_amount': 100000,
                'credit_score': 580,
                'loan_purpose': 'Personal',
                'debt_to_income': 0.65
            }
        ]
        
        # Process each test case
        for case in test_cases:
            print(f"\n📋 {case['name']}:")
            print("-" * 30)
            
            try:
                prediction, probabilities, info = predictor.predict(
                    age=case['age'],
                    income=case['income'],
                    years_employed=case['years_employed'],
                    loan_amount=case['loan_amount'],
                    credit_score=case['credit_score'],
                    loan_purpose=case['loan_purpose'],
                    debt_to_income=case['debt_to_income']
                )
                
                result = "✅ APPROVED" if prediction == 1 else "❌ DENIED"
                probability = probabilities[1] if prediction == 1 else probabilities[0]
                
                print(f"Result: {result}")
                print(f"Confidence: {probability:.1%}")
                print(f"Risk Level: {info['risk_level']}")
                
                if info['recommendations']:
                    print("💡 Recommendations:")
                    for rec in info['recommendations']:
                        print(f"   • {rec}")
                        
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")

if __name__ == "__main__":
    demo_predictions()