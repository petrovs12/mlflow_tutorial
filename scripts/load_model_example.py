import mlflow
from typing import Any

def load_and_predict() -> None:
    """Load the latest model from MLflow and make a prediction"""
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Get the latest run
    latest_run = mlflow.search_runs(
        experiment_names=["retail_prediction"],
        order_by=["start_time DESC"],
        max_results=1
    ).iloc[0]
    
    # Load the model
    model_uri = f"runs:/{latest_run.run_id}/model"
    loaded_model: Any = mlflow.sklearn.load_model(model_uri)
    
    # Make a prediction with the example data
    example_data = {
        "transaction_amount": [100.0],
        "num_items": [5],
        "seasonal_factor": [0.5]
    }
    
    prediction = loaded_model.predict([[100.0, 5, 0.5]])
    print(f"Prediction for example data: {prediction}")

if __name__ == "__main__":
    load_and_predict() 