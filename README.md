# MLOps Project with Feast and MLflow

This project demonstrates a machine learning workflow using Feast for feature management and MLflow for experiment tracking.

## Project Setup

### 1. Development Environment Setup

The project uses VS Code Dev Containers for a consistent development environment. The container includes:
- Python environment with required packages
- Redis for Feast online store
- MLflow tracking server

To start:
1. Open the project in VS Code
2. Install the "Remote - Containers" extension
3. Click "Reopen in Container" when prompted
4. The container will automatically:
   - Install all dependencies
   - Start Redis server on port 6379

### 2. Feature Store Setup

The project uses Feast for feature management. Here's how to set up the feature store:

```bash
# Generate synthetic data
python scripts/generate_data.py

# Apply Feast configuration
cd feature_store
feast apply
cd ..

# Test Feast setup
python scripts/test_feast.py
```

The data generation creates:
- Recent timestamps (last 3 days)
- Customer transactions
- Store information
- Target variables for prediction

### 3. MLflow Setup

MLflow is used for experiment tracking. To start:

```bash
# Start MLflow server (in a separate terminal)
python scripts/manage_mlflow.py

# To stop MLflow server if needed
python scripts/manage_mlflow.py stop
```

The MLflow UI will be available at: http://localhost:5000

### 4. Training and Experiment Tracking

To train models and track experiments:

```bash
# Train a single model
python scripts/train_model.py

# Run multiple experiments (10 runs with different parameters)
python scripts/run_experiments.py
```

To load and use a trained model:
```bash
python scripts/load_model.py
```

## Project Structure

```
project_root/
├── .devcontainer/              # Development container configuration
│   ├── Dockerfile             # Container definition
│   └── devcontainer.json      # Dev container settings
├── feature_store/             # Feast configuration
│   ├── data/                  # Generated feature data
│   ├── feature_store.yaml     # Feast configuration
│   └── features.py           # Feature definitions
├── scripts/
│   ├── generate_data.py      # Data generation script
│   ├── test_feast.py         # Feast testing script
│   ├── manage_mlflow.py      # MLflow server management
│   ├── train_model.py        # Model training script
│   ├── run_experiments.py    # Multiple experiment runner
│   └── load_model.py         # Model loading utility
└── requirements.txt          # Python dependencies
```

## Key Features

1. **Feature Store (Feast)**
   - Online and offline feature serving
   - Redis-based online store
   - File-based offline store
   - Recent data with 3-day window

2. **Experiment Tracking (MLflow)**
   - Model metrics tracking
   - Feature importance visualization
   - Cross-validation results
   - Model artifacts storage
   - Easy model loading and serving

3. **Model Training**
   - RandomForest classifier
   - Cross-validation
   - Feature importance analysis
   - Multiple experiment runs

## Notes

- The Redis server is automatically started in the development environment
- MLflow server must be running to track experiments
- Feature data is generated with recent timestamps for proper online serving
- All scripts use proper type hints and error handling

