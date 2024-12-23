import subprocess
import os
from pathlib import Path

def start_mlflow_server() -> None:
    """Start MLflow server with the correct configuration"""
    # Create mlruns directory if it doesn't exist
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    
    # Command to start MLflow
    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--serve-artifacts",
        "--workers", "2",
        "--enable-cors"
    ]
    
    # Start the server
    subprocess.run(cmd)

if __name__ == "__main__":
    start_mlflow_server() 