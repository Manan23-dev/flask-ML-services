from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Simple preprocessing function (since we can't import from app.utils yet)
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(0)
    
    # Required columns
    required_columns = ['user_id', 'product_category', 'price', 'rating', 'purchase_count']
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only required columns
    df = df[required_columns]
    
    # Encode categorical variables
    if 'product_category' in df.columns:
        category_mapping = {
            'electronics': 1, 'clothing': 2, 'books': 3, 
            'home': 4, 'sports': 5, 'other': 0
        }
        df['product_category'] = df['product_category'].map(category_mapping).fillna(0)
    
    return df.values

# Load model (will be created after training)
model = None
try:
    model = joblib.load("./models/latest_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model not found: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'Flask ML Service'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data).max()
        
        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': 'product_recommender',
        'version': '1.0',
        'status': 'active' if model is not None else 'not_loaded'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
