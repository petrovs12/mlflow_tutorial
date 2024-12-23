import pandas as pd
from feast import FeatureStore

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the ingested data.

    Args:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Data with engineered features.
    """
    # Example feature engineering
    data['feature_1'] = data['column_a'] * 2
    data['feature_2'] = data['column_b'].apply(lambda x: x.lower())
    return data

def main():
    input_path = 'data/raw/data.csv'
    data = pd.read_csv(input_path)
    engineered_data = create_features(data)

    # Initialize Feast Feature Store
    store = FeatureStore(repo_path=".")

    # Register features with Feast
    # Example:
    # store.apply([feature_table])

    print("Feature engineering completed.")

if __name__ == "__main__":
    main()
