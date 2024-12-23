import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
import joblib

def train_sklearn_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/sklearn_model.pkl')
    print("Scikit-Learn model trained and saved.")

def train_statsmodels_model(X_train, y_train):
    X_train_const = add_constant(X_train)
    model = OLS(y_train, X_train_const).fit()
    model.save('models/statsmodels_model.pkl')
    print("StatsModels model trained and saved.")

def main():
    data_path = 'data/processed/engineered_data.csv'
    data = pd.read_csv(data_path)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_sklearn_model(X_train, y_train)
    train_statsmodels_model(X_train, y_train)

if __name__ == "__main__":
    main()
