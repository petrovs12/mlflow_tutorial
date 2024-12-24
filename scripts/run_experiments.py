import subprocess
from typing import List
import time

def run_experiments(n_runs: int = 10) -> None:
    """Run multiple training experiments"""
    for i in range(n_runs):
        print(f"\nStarting Run {i}")
        subprocess.run(["python", "scripts/train_model.py", str(i)])
        time.sleep(1)  # Small delay between runs

if __name__ == "__main__":
    run_experiments() 