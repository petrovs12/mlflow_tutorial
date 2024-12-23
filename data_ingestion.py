import pandas as pd
from feast import FeatureStore

def ingest_data(input_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file.

    Args:
        input_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Ingested data as a DataFrame.
    """
    data = pd.read_csv(input_path)
    return data

def main():
    input_path = 'data/raw/data.csv'
    data = ingest_data(input_path)
    # Initialize Feast Feature Store
    store = FeatureStore(repo_path=".")

    # Define your features in Feast here
    # Example:
    # store.apply([feature_table])

    print("Data ingestion completed.")

if __name__ == "__main__":
    main()
