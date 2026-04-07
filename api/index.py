from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for local testing without issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globally
model = None
scaler = None
label_encoder = None
feature_names = None

try:
    # Vercel gives paths relative to root or api folder depending on deployment.
    potential_paths = [
        os.path.join(os.path.dirname(__file__), 'credit_score_ai_model.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'credit_score_ai_model.pkl'),
        os.path.join(os.getcwd(), 'credit_score_ai_model.pkl'),
        'credit_score_ai_model.pkl'
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            model_data = joblib.load(path)
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
            feature_names = model_data.get('feature_names', [])
            print(f"Loaded model successfully from {path}")
            break
            
    if model is None:
        print("Failed to find credit_score_ai_model.pkl in expected paths.")
        
except Exception as e:
    print(f"Error loading model: {e}")

class PredictRequest(BaseModel):
    age: int
    income: float
    years_employed: int
    loan_amount: float
    credit_score: int
    loan_purpose: str
    debt_to_income: float

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model failed to load on the server.")
    
    try:
        try:
            loan_purpose_encoded = label_encoder.transform([req.loan_purpose])[0]
        except ValueError:
            loan_purpose_encoded = 0

        features = np.array([[
            req.age, req.income, req.years_employed, req.loan_amount,
            req.credit_score, loan_purpose_encoded, req.debt_to_income
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0].tolist()
        
        # Return results
        importances = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else []
        
        return {
            "prediction": prediction,
            "probability": probability,
            "feature_importances": importances,
            "features": feature_names if feature_names else ["Age", "Income", "Years Employed", "Loan Amount", "Credit Score", "Loan Purpose", "Debt-to-Income"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
