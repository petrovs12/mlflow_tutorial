from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import pandas as pd
from feast import FeatureStore

def test_feast_store() -> None:
    # Get absolute path to feature store directory
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    feature_store_path: str = os.path.join(current_dir, "..", "feature_store")
    
    # Initialize the feature store
    store: FeatureStore = FeatureStore(repo_path=feature_store_path)
    
    # Get historical features
    entity_df: pd.DataFrame = pd.DataFrame.from_dict({
        "customer_id": [1, 2, 3],
        "store_id": [1, 2, 3],
        "event_timestamp": [
            datetime.now(),
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=2)
        ]
    })
    
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
    
    # Test online store
    store.materialize_incremental(end_date=datetime.now())
    
    online_features: Dict[str, List[Any]] = store.get_online_features(
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:num_items",
            "transaction_stats:seasonal_factor",
        ],
        entity_rows=[{"customer_id": 1, "store_id": 1}],
    ).to_dict()
    
    print("\nRetrieved Online Features:")
    print(online_features)

if __name__ == "__main__":
    test_feast_store() 