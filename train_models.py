import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_sklearn_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/sklearn_model.pkl')
    print("Scikit-Learn model trained and saved.")

def main():
    # Create simple training data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples)
    })
    y = 2 * X['feature_1'] + X['feature_2'] + np.random.randn(n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_sklearn_model(X_train, y_train)

if __name__ == "__main__":
    main()
