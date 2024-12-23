from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import pandas as pd
from feast import FeatureStore

def test_feast_store() -> None:
    """Test Feast feature retrieval with actual data timestamps"""
    # Get absolute path to feature store directory
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    feature_store_path: str = os.path.join(current_dir, "..", "feature_store")
    
    # Read actual data to get real timestamps
    data_path: str = os.path.join(feature_store_path, "data", "train_data.parquet")
    actual_data: pd.DataFrame = pd.read_parquet(data_path)
    
    # Get a sample of real timestamps from our data
    sample_timestamps = actual_data['event_timestamp'].head(3).tolist()
    sample_customers = actual_data['customer_id'].head(3).tolist()
    sample_stores = actual_data['store_id'].head(3).tolist()
    
    # Create entity DataFrame with actual timestamps
    entity_df: pd.DataFrame = pd.DataFrame({
        "customer_id": sample_customers,
        "store_id": sample_stores,
        "event_timestamp": sample_timestamps
    })
    
    # Initialize the feature store
    store: FeatureStore = FeatureStore(repo_path=feature_store_path)
    
    print("\nEntity DataFrame:")
    print(entity_df)
    
    # Get historical features
    training_df: pd.DataFrame = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:num_items",
            "transaction_stats:seasonal_factor",
        ],
    ).to_df()
    
    print("\nRetrieved Historical Features:")
    print(training_df.head())
    
    # Materialize features to online store
    print("\nMaterializing features to online store...")
    store.materialize(
        start_date=datetime.now() - timedelta(days=3),
        end_date=datetime.now(),
    )
    
    # Use actual values for online test
    latest_timestamp = actual_data['event_timestamp'].max()
    online_features: Dict[str, List[Any]] = store.get_online_features(
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:num_items",
            "transaction_stats:seasonal_factor",
        ],
        entity_rows=[{
            "customer_id": sample_customers[0],
            "store_id": sample_stores[0]
        }],
    ).to_dict()
    
    print("\nRetrieved Online Features:")
    print(online_features)

if __name__ == "__main__":
    test_feast_store() 