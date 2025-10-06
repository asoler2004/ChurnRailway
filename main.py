from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
from typing import Optional

# Load the trained model and metadata
try:
    model = joblib.load('churn_model.pkl')
    encoders = joblib.load('encoders.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']
    use_onehot = metadata['use_onehot']
    encoder_metadata = metadata['encoders']
except FileNotFoundError:
    raise Exception("Model files not found. Please run train_model.py first.")

app = FastAPI(title="Churn Prediction API", version="1.0.0")

class CustomerData(BaseModel):
    age: int
    housing: str
    credit_score: float
    deposits: int
    withdrawal: int
    purchases_partners: int
    purchases: int
    cc_taken: int
    cc_recommended: int
    cc_disliked: int
    cc_liked: int
    cc_application_begin: int
    app_downloaded: int
    web_user: int
    app_web_user: int
    ios_user: int
    android_user: int
    registered_phones: int
    payment_type: str
    waiting_4_loan: int
    cancelled_loan: int
    received_loan: int
    rejected_loan: int
    zodiac_sign: str
    left_for_two_month_plus: int
    left_for_one_month: int
    rewards_earned: int
    reward_rate: float
    is_referred: int

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str

def preprocess_input(customer_data: CustomerData):
    """Preprocess input data to match training format"""
    # Convert to dictionary
    data_dict = customer_data.dict()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Handle categorical variables
    categorical_columns = ['housing', 'payment_type', 'zodiac_sign']
    
    if use_onehot:
        # Use OneHotEncoder
        for col in categorical_columns:
            if col in df.columns and col in encoders:
                encoder = encoders[col]
                try:
                    # Transform the categorical column
                    encoded_data = encoder.transform(df[[col]].astype(str))
                    
                    # Create column names for one-hot encoded features
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0][1:]]  # Skip first due to drop='first'
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                    
                    # Drop original column and add encoded columns
                    df = df.drop(col, axis=1)
                    df = pd.concat([df, encoded_df], axis=1)
                    
                except Exception:
                    # Handle unseen categories - encoder will handle this with handle_unknown='ignore'
                    # Create zero columns for this categorical variable
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0][1:]]
                    zero_df = pd.DataFrame(0, columns=feature_names, index=df.index)
                    df = df.drop(col, axis=1)
                    df = pd.concat([df, zero_df], axis=1)
    else:
        # Use LabelEncoder
        for col in categorical_columns:
            if col in df.columns and col in encoder_metadata:
                classes = encoder_metadata[col]['classes']
                value = str(df[col].iloc[0])
                
                if value in classes:
                    df[col] = classes.index(value)
                else:
                    # Handle unseen categories by assigning to most frequent class (index 0)
                    df[col] = 0
    
    # Ensure all feature columns are present and in correct order
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_columns]
    
    return df

@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """Predict churn probability for a customer"""
    try:
        # Preprocess the input data
        processed_data = preprocess_input(customer_data)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0][1]
        churn_prediction = int(churn_probability > 0.5)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "feature_columns": feature_columns,
        "encoding_type": "OneHot" if use_onehot else "Label",
        "categorical_encodings": encoder_metadata,
        "model_type": "XGBoost Classifier"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)