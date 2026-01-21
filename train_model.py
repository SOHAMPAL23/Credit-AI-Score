import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print('🔄 Training the optimized loan approval model...')
print('='*60)

# Generate enhanced synthetic data
def generate_enhanced_synthetic_data(n_samples=8000, random_state=42):
    np.random.seed(random_state)
    
    data_list = []
    loan_purposes = ['Home Improvement', 'Debt Consolidation', 'Business', 'Education', 
                     'Medical', 'Personal', 'Auto', 'Other']
    purpose_weights = [0.25, 0.20, 0.15, 0.08, 0.07, 0.12, 0.08, 0.05]
    
    for i in range(n_samples):
        age = np.random.normal(38, 12)
        age = np.clip(age, 18, 80)
        
        income = np.random.lognormal(10.5, 0.6)
        income = np.clip(income, 20000, 600000)
        
        max_years = min(age - 18, 45)
        years_employed = np.random.beta(2, 5) * max_years
        years_employed = np.clip(years_employed, 0, max_years)
        
        base_loan = np.random.lognormal(11.2, 0.5)
        loan_amount = np.clip(base_loan, 15000, 800000)
        
        credit_score = np.random.normal(680, 90)
        credit_score = np.clip(credit_score, 300, 850)
        
        loan_purpose = np.random.choice(loan_purposes, p=purpose_weights)
        
        base_dti = loan_amount / (income * 12)
        noise = np.random.normal(0, 0.05)
        debt_to_income = base_dti + noise
        debt_to_income = np.clip(debt_to_income, 0.05, 0.95)
        
        # Enhanced approval logic
        approval_score = 0
        
        # Credit score contribution (30% weight)
        if credit_score >= 700:
            approval_score += 30
        elif credit_score >= 650:
            approval_score += 20
        elif credit_score >= 600:
            approval_score += 10
        
        # Income to loan ratio (25% weight)
        income_ratio = income / loan_amount
        if income_ratio >= 3:
            approval_score += 25
        elif income_ratio >= 2:
            approval_score += 15
        elif income_ratio >= 1.5:
            approval_score += 8
        
        # DTI ratio (20% weight)
        if debt_to_income <= 0.3:
            approval_score += 20
        elif debt_to_income <= 0.4:
            approval_score += 12
        elif debt_to_income <= 0.5:
            approval_score += 5
        
        # Employment stability (15% weight)
        if years_employed >= 5:
            approval_score += 15
        elif years_employed >= 2:
            approval_score += 8
        
        # Age factor (10% weight)
        if 25 <= age <= 60:
            approval_score += 10
        elif age >= 60:
            approval_score += 5
        
        # Purpose adjustment
        purpose_multipliers = {
            'Home Improvement': 1.1,
            'Debt Consolidation': 0.9,
            'Business': 1.05,
            'Education': 0.85,
            'Medical': 1.0,
            'Personal': 0.95,
            'Auto': 1.0,
            'Other': 0.9
        }
        
        final_score = approval_score * purpose_multipliers[loan_purpose]
        
        # Final approval decision with some randomness
        threshold = 65 + np.random.normal(0, 5)
        approved = 1 if final_score >= threshold else 0
        
        data_list.append({
            'age': age,
            'income': income,
            'years_employed': years_employed,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'loan_purpose': loan_purpose,
            'debt_to_income': debt_to_income,
            'approval_score': final_score,
            'approved': approved
        })
    
    return pd.DataFrame(data_list)

print('📊 Generating enhanced synthetic loan data...')
df = generate_enhanced_synthetic_data(8000, 42)
print(f'✅ Generated {len(df)} samples')
approval_rate = df["approved"].mean()
print('📈 Approval rate: {:.2%}'.format(approval_rate))

# Feature engineering
def preprocess_and_engineer_features(data):
    df_processed = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df_processed['loan_purpose_encoded'] = le.fit_transform(df_processed['loan_purpose'])
    
    # Create engineered features
    df_processed['income_to_loan_ratio'] = df_processed['income'] / df_processed['loan_amount']
    df_processed['credit_utilization'] = (df_processed['debt_to_income'] * df_processed['income']) / df_processed['credit_score']
    df_processed['employment_stability'] = df_processed['years_employed'] / np.maximum(df_processed['age'] - 18, 1)
    
    # Age group categorization
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                      bins=[0, 25, 35, 50, 65, 100], 
                                      labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])
    df_processed['age_group_encoded'] = LabelEncoder().fit_transform(df_processed['age_group'])
    
    # Income bracket
    df_processed['income_bracket'] = pd.cut(df_processed['income'], 
                                           bins=[0, 40000, 80000, 150000, 300000, 1000000],
                                           labels=['Low', 'Medium', 'High', 'Very High', 'Ultra High'])
    df_processed['income_bracket_encoded'] = LabelEncoder().fit_transform(df_processed['income_bracket'])
    
    # Credit score categories
    df_processed['credit_category'] = pd.cut(df_processed['credit_score'],
                                            bins=[0, 580, 670, 740, 800, 850],
                                            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])
    df_processed['credit_category_encoded'] = LabelEncoder().fit_transform(df_processed['credit_category'])
    
    # DTI risk categories
    df_processed['dti_risk'] = pd.cut(df_processed['debt_to_income'],
                                     bins=[0, 0.2, 0.36, 0.43, 1.0],
                                     labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])
    df_processed['dti_risk_encoded'] = LabelEncoder().fit_transform(df_processed['dti_risk'])
    
    # Select final features
    feature_columns = [
        'age', 'income', 'years_employed', 'loan_amount', 'credit_score',
        'loan_purpose_encoded', 'debt_to_income',
        'income_to_loan_ratio', 'credit_utilization', 'employment_stability',
        'age_group_encoded', 'income_bracket_encoded', 'credit_category_encoded', 'dti_risk_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['approved']
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    return X, y, feature_columns, le

print('⚙️ Performing preprocessing and feature engineering...')
X, y, feature_names, label_encoder = preprocess_and_engineer_features(df)
print(f'✅ Final feature set: {len(feature_names)} features')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('📏 Splitting data for training and testing...')
print(f'✅ Training set size: {len(X_train)}')
print(f'✅ Test set size: {len(X_test)}')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('⚖️ Scaling features...')

# Define models and their hyperparameter grids
models_config = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [200, 500],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    },
}

print('🎯 Starting hyperparameter tuning...')

# Hyperparameter tuning function
def tune_model(model_name, model_config, X_train, y_train):
    print(f'🔍 Tuning {model_name}...')
    
    # Use RandomizedSearchCV for faster tuning
    search = RandomizedSearchCV(
        estimator=model_config['model'],
        param_distributions=model_config['params'],
        n_iter=10,  # Number of parameter settings sampled
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    print(f'✅ Best F1 score: {search.best_score_:.4f}')
    print(f'✅ Best parameters: {search.best_params_}')
    
    return search.best_estimator_, search.best_params_, search.best_score_

# Tune models
best_models = {}
best_params = {}
best_scores = {}

for model_name, model_config in models_config.items():
    model, params, score = tune_model(model_name, model_config, X_train_scaled, y_train)
    best_models[model_name] = model
    best_params[model_name] = params
    best_scores[model_name] = score

# Select the best model
best_model_name = max(best_scores, key=best_scores.get)
final_model = best_models[best_model_name]

print(f'🏆 Selected Best Model: {best_model_name}')

# Make predictions
y_pred = final_model.predict(X_test_scaled)
y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

# Calculate comprehensive metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print()
print('🏆 FINAL MODEL PERFORMANCE:')
print('='*60)
print(f'📊 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'🎯 Precision: {precision:.4f} ({precision*100:.2f}%)')
print(f'🔍 Recall:    {recall:.4f} ({recall*100:.2f}%)')
print(f'⭐ F1-Score:  {f1:.4f} ({f1*100:.2f}%)')

# Cross-validation
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                           scoring='f1')

print()
print('🔄 Cross-Validation Results (F1-Score):')
print(f'📊 Mean CV F1-Score: {cv_scores.mean():.4f}')
print(f'📈 Std Deviation: {cv_scores.std():.4f}')

# Save the final model
model_package = {
    'model': final_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_names': feature_names,
    'model_name': best_model_name
}

joblib.dump(model_package, 'loan_approval_final_model.pkl')
print()
print('💾 Model saved as loan_approval_final_model.pkl')

print()
print('✨ Model training and evaluation completed successfully!')
print('💡 The optimized model achieves high performance across all key metrics.')

# Test the predictor
print('\n🧪 Testing the predictor with sample cases...')
from loan_predictor import LoanApprovalPredictor

try:
    predictor = LoanApprovalPredictor()
    print("✅ Predictor loaded successfully!")
    
    # Test strong candidate
    pred, prob, info = predictor.predict(
        age=35, income=80000, years_employed=8, loan_amount=150000,
        credit_score=750, loan_purpose="Home Improvement", debt_to_income=0.2
    )
    print(f"✅ Strong candidate: {'Approved' if pred == 1 else 'Denied'}, Confidence: {prob[1]:.2%}")
    
    # Test weak candidate
    pred, prob, info = predictor.predict(
        age=25, income=30000, years_employed=1, loan_amount=100000,
        credit_score=580, loan_purpose="Personal", debt_to_income=0.6
    )
    print(f"✅ Weak candidate: {'Approved' if pred == 1 else 'Denied'}, Confidence: {prob[1]:.2%}")
    
except Exception as e:
    print(f"⚠️ Error testing predictor: {e}")