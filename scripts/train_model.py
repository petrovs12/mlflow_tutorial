import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
from feast import FeatureStore

def get_training_data(store: FeatureStore) -> pd.DataFrame:
    """Get historical features from Feast"""
    # Read the raw data to get timestamps and entity values
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_path: str = os.path.join(script_dir, "..", "feature_store", "data", "train_data.parquet")
    entity_df: pd.DataFrame = pd.read_parquet(data_path)
    
    # Get historical features from Feast
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:num_items",
            "transaction_stats:seasonal_factor",
        ],
    ).to_df()
    
    return training_df

def train_model(training_df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, float], pd.DataFrame, pd.Series]:
    """Train a model using the features"""
    # Prepare features and target
    feature_columns = [
        "transaction_amount",
        "num_items",
        "seasonal_factor"
    ]
    
    X: pd.DataFrame = training_df[feature_columns]
    y: pd.Series = training_df["target"]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get validation predictions
    val_predictions = model.predict(X_val)
    
    # Log metrics with MLflow
    metrics: Dict[str, float] = {
        "accuracy": model.score(X_val, y_val),
    }
    
    return model, metrics, X_val, y_val

def main() -> None:
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("retail_prediction")
    
    # Initialize feature store
    repo_path: str = os.path.join(os.path.dirname(__file__), "..", "feature_store")
    store: FeatureStore = FeatureStore(repo_path=repo_path)
    
    # Get training data
    training_df: pd.DataFrame = get_training_data(store)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model and get metrics
        model, metrics, X_val, y_val = train_model(training_df)
        
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Create an input example
        input_example = pd.DataFrame({
            "transaction_amount": [100.0],
            "num_items": [5],
            "seasonal_factor": [0.5]
        })
        
        # Log model with input example
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=input_example,
            signature=mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
        )
        
        # Print results
        print("\nTraining Results:")
        print(f"Metrics: {metrics}")
        print("\nClassification Report:")
        print(classification_report(y_val, model.predict(X_val)))
        
        print("\nModel Input Example:")
        print(input_example)

if __name__ == "__main__":
    main() 