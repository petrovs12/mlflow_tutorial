import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import mlflow
from feast import FeatureStore

def get_training_data(store: FeatureStore) -> pd.DataFrame:
    """Get historical features from Feast"""
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_path: str = os.path.join(script_dir, "..", "feature_store", "data", "train_data.parquet")
    entity_df: pd.DataFrame = pd.read_parquet(data_path)
    
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:num_items",
            "transaction_stats:seasonal_factor",
        ],
    ).to_df()
    
    training_df['target'] = entity_df['target'].values
    return training_df

def train_model(training_df: pd.DataFrame, n_estimators: int = 100) -> Tuple[RandomForestClassifier, Dict[str, float], pd.DataFrame, pd.Series]:
    """Train a model using the features with cross-validation"""
    feature_columns = [
        "transaction_amount",
        "num_items",
        "seasonal_factor"
    ]
    
    X: pd.DataFrame = training_df[feature_columns]
    y: pd.Series = training_df["target"]
    
    # Cross validation
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Train final model
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    val_predictions = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_predictions, average='binary')
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log metrics
    metrics: Dict[str, float] = {
        "accuracy": model.score(X_val, y_val),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return model, metrics, X_val, y_val, feature_importance

def main(run_id: int = 0) -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("retail_prediction")
    
    repo_path: str = os.path.join(os.path.dirname(__file__), "..", "feature_store")
    store: FeatureStore = FeatureStore(repo_path=repo_path)
    
    training_df: pd.DataFrame = get_training_data(store)
    
    # Vary n_estimators for different runs
    n_estimators = 50 + run_id * 50  # 50, 100, 150, ...
    
    with mlflow.start_run(run_name=f"run_{run_id}"):
        model, metrics, X_val, y_val, feature_importance = train_model(training_df, n_estimators)
        
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "run_id": run_id
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log feature importance as a figure
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        # Create and log input example
        input_example = pd.DataFrame({
            "transaction_amount": [100.0],
            "num_items": [5],
            "seasonal_factor": [0.5]
        })
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=input_example,
            signature=mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
        )
        
        print(f"\nRun {run_id} Results:")
        print(f"Metrics: {metrics}")
        print("\nFeature Importance:")
        print(feature_importance)
        print("\nClassification Report:")
        print(classification_report(y_val, model.predict(X_val)))

if __name__ == "__main__":
    import sys
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(run_id) 