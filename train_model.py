import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import joblib
import json

def load_and_preprocess_data(csv_file_path, use_onehot=False):
    """Load and preprocess the churn data
    
    Args:
        csv_file_path: Path to the CSV file
        use_onehot: If True, use OneHotEncoder; if False, use LabelEncoder
    """
    df = pd.read_csv(csv_file_path)
    
    # Remove user column as it's just an ID
    if 'user' in df.columns:
        df = df.drop('user', axis=1)
    
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Handle categorical variables
    categorical_columns = ['housing', 'payment_type', 'zodiac_sign']
    encoders = {}
    
    if use_onehot:
        # Use OneHotEncoder
        for col in categorical_columns:
            if col in X.columns:
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_data = ohe.fit_transform(X[[col]].astype(str))
                
                # Create column names for one-hot encoded features
                feature_names = [f"{col}_{category}" for category in ohe.categories_[0][1:]]  # Skip first due to drop='first'
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                
                # Drop original column and add encoded columns
                X = X.drop(col, axis=1)
                X = pd.concat([X, encoded_df], axis=1)
                
                encoders[col] = ohe
    else:
        # Use LabelEncoder
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
    
    return X, y, encoders, use_onehot

def train_xgboost_model(X, y):
    """Train XGBoost classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X.columns.tolist()

def save_model_and_metadata(model, feature_columns, encoders, use_onehot):
    """Save the trained model and preprocessing metadata"""
    # Save the model
    joblib.dump(model, 'churn_model.pkl')
    
    # Save encoders
    joblib.dump(encoders, 'encoders.pkl')
    
    # Save feature columns and encoding metadata
    if use_onehot:
        # For OneHotEncoder, save categories
        encoder_metadata = {}
        for col, encoder in encoders.items():
            encoder_metadata[col] = {
                'type': 'onehot',
                'categories': [list(cat) for cat in encoder.categories_]
            }
    else:
        # For LabelEncoder, save classes
        encoder_metadata = {}
        for col, encoder in encoders.items():
            encoder_metadata[col] = {
                'type': 'label',
                'classes': list(encoder.classes_)
            }
    
    metadata = {
        'feature_columns': feature_columns,
        'use_onehot': use_onehot,
        'encoders': encoder_metadata
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model and metadata saved successfully!")

if __name__ == "__main__":
    # Configuration
    csv_file = "churn_data.csv"  # Update this path to your CSV file
    use_onehot = True  # Set to True to use OneHotEncoder instead of LabelEncoder
    
    try:
        X, y, encoders, encoding_type = load_and_preprocess_data(csv_file, use_onehot=use_onehot)
        print(f"Data loaded successfully. Shape: {X.shape}")
        print(f"Encoding method: {'OneHot' if encoding_type else 'Label'}")
        
        # Train the model
        model, feature_columns = train_xgboost_model(X, y)
        
        # Save model and metadata
        save_model_and_metadata(model, feature_columns, encoders, encoding_type)
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please ensure your CSV file is in the same directory and named 'churn_data.csv'")
    except Exception as e:
        print(f"Error during training: {str(e)}")