import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_samples: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic time series data for ML training
    
    Returns:
    - Training data (80% of samples)
    - Test data (20% of samples)
    """
    # Generate timestamps
    end_date: datetime = datetime.now()
    start_date: datetime = end_date - timedelta(days=n_samples)
    dates: pd.DatetimeIndex = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate feature data
    data: dict = {
        'timestamp': dates,
        'customer_id': np.random.randint(1, 100, n_samples),
        'transaction_amount': np.random.normal(100, 30, n_samples),
        'num_items': np.random.randint(1, 10, n_samples),
        'store_id': np.random.randint(1, 50, n_samples),
    }
    
    # Add some seasonal patterns
    data['seasonal_factor'] = np.sin(2 * np.pi * np.arange(n_samples) / 365.25) * 10
    
    # Generate target variable (e.g., customer will make purchase next week)
    # Target is influenced by transaction_amount, num_items, and seasonal_factor
    target = (
        0.3 * data['transaction_amount'] / 100 +
        0.2 * data['num_items'] +
        0.1 * data['seasonal_factor'] +
        np.random.normal(0, 1, n_samples)
    )
    data['target'] = (target > target.mean()).astype(int)
    
    # Create DataFrame
    df: pd.DataFrame = pd.DataFrame(data)
    
    # Split into train and test
    train_size: int = int(0.8 * n_samples)
    train_df: pd.DataFrame = df[:train_size]
    test_df: pd.DataFrame = df[train_size:]
    
    return train_df, test_df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save the generated data to Parquet files in the feature store data directory"""
    # Get the script directory
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    # Create feature store data directory
    feature_store_data_dir: str = os.path.join(script_dir, "..", "feature_store", "data")
    os.makedirs(feature_store_data_dir, exist_ok=True)
    
    # Save files as Parquet
    train_path: str = os.path.join(feature_store_data_dir, "train_data.parquet")
    test_path: str = os.path.join(feature_store_data_dir, "test_data.parquet")
    
    # Add created timestamp for Feast
    train_df['created'] = pd.Timestamp.now()
    test_df['created'] = pd.Timestamp.now()
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Data saved to {feature_store_data_dir}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    # Generate data
    train_df, test_df = generate_synthetic_data(n_samples=10000)
    
    # Save to files
    save_data(train_df, test_df)
    
    # Print sample of the data
    print("\nSample of training data:")
    print(train_df.head())
    
    # Print basic statistics
    print("\nData Statistics:")
    print(train_df.describe()) 