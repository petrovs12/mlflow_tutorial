import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
import pandas as pd
from sklearn.model_selection import train_test_split

def train_and_track_sklearn(X_train, y_train):
    with mlflow.start_run(run_name="Sklearn_LinearRegression") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("coef_", model.coef_[0])
        print("Scikit-Learn model trained and tracked with MLflow.")

def train_and_track_statsmodels(X_train, y_train):
    with mlflow.start_run(run_name="StatsModels_OLS") as run:
        X_train_const = add_constant(X_train)
        model = OLS(y_train, X_train_const).fit()
        mlflow.statsmodels.log_model(model, "model")
        mlflow.log_param("model_type", "OLS")
        mlflow.log_metric("R_squared", model.rsquared)
        print("StatsModels model trained and tracked with MLflow.")

def main():
    data_path = 'data/processed/engineered_data.csv'
    data = pd.read_csv(data_path)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MLOps_Project_Experiment")

    train_and_track_sklearn(X_train, y_train)
    train_and_track_statsmodels(X_train, y_train)

if __name__ == "__main__":
    main()
