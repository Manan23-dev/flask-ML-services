import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib
import os

def preprocess_data(df):
    """Preprocess input data for model prediction"""
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

def generate_sample_data(n_samples=1000):
    """Generate sample e-commerce data"""
    np.random.seed(42)
    
    data = {
        'user_id': np.random.randint(1, 100, n_samples),
        'product_category': np.random.choice(['electronics', 'clothing', 'books', 'home', 'sports'], n_samples),
        'price': np.random.uniform(10, 500, n_samples),
        'rating': np.random.uniform(1, 5, n_samples),
        'purchase_count': np.random.randint(1, 20, n_samples),
        'will_recommend': np.random.randint(0, 2, n_samples)  # Target variable
    }
    
    return pd.DataFrame(data)

def train_model():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Set MLflow experiment
    mlflow.set_experiment("product_recommender")
    
    with mlflow.start_run():
        print("🚀 Starting model training...")
        
        # Generate sample data
        df = generate_sample_data(2000)
        print(f"📊 Generated {len(df)} sample records")
        
        # Prepare features and target
        X = df.drop('will_recommend', axis=1)
        y = df['will_recommend']
        
        # Preprocess features
        X_processed = preprocess_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📈 Model accuracy: {accuracy:.4f}")
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model with MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="product_recommender"
        )
        
        # Save model locally as backup
        joblib.dump(model, "./models/latest_model.pkl")
        
        print("✅ Model training completed successfully!")
        print(f"📁 Model saved to ./models/latest_model.pkl")
        print(f"📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        return model, accuracy

if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"\n🎯 Final model accuracy: {accuracy:.4f}")
