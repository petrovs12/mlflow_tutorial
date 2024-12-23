import subprocess
import os
from pathlib import Path
from typing import List

def get_mlflow_processes() -> List[str]:
    """Get list of running MLflow process IDs"""
    try:
        # Find all python processes running mlflow
        ps = subprocess.run(
            ["ps", "aux", "|", "grep", "mlflow", "|", "grep", "-v", "grep"],
            shell=True,
            capture_output=True,
            text=True
        )
        return [line.split()[1] for line in ps.stdout.splitlines() if "mlflow" in line]
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []

def stop_mlflow_servers() -> None:
    """Stop all running MLflow servers"""
    pids = get_mlflow_processes()
    if not pids:
        print("No MLflow servers found running")
        return
    
    print(f"Found {len(pids)} MLflow processes")
    for pid in pids:
        try:
            subprocess.run(["kill", pid])
            print(f"Stopped MLflow server with PID {pid}")
        except Exception as e:
            print(f"Error stopping process {pid}: {e}")

def start_mlflow_server() -> None:
    """Start MLflow server with the correct configuration"""
    # First stop any running servers
    stop_mlflow_servers()
    
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
        "--workers", "2"
    ]
    
    # Start the server
    print("Starting new MLflow server...")
    subprocess.run(cmd)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_mlflow_servers()
    else:
        start_mlflow_server() 