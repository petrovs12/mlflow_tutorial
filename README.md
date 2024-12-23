# MLOps Project

This repository demonstrates best practices in Machine Learning Operations (MLOps) using tools like Feast, MLflow, and Seldon Core. The project is containerized with Docker to ensure reproducibility and ease of deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Data Ingestion and Feature Engineering](#data-ingestion-and-feature-engineering)
- [Model Training and Experiment Tracking](#model-training-and-experiment-tracking)
- [Model Deployment](#model-deployment)
- [Running with Docker](#running-with-docker)
- [Documentation](#documentation)

## Prerequisites

- Python 3.9+
- Docker
- Kubernetes cluster (for Seldon Core)
- MLflow server
- Feast Feature Store

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/mlops-project.git
    cd mlops-project
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure MLflow:**

    Ensure that the MLflow tracking server is running and update the `mlflow_tracking.py` with the correct tracking URI.

4. **Configure Feast:**

    Initialize and apply Feast configurations.

    ```bash
    feast init
    feast apply
    ```

## Data Ingestion and Feature Engineering

1. **Ingest Data:**

    ```bash
    python data_ingestion.py
    ```

2. **Feature Engineering:**

    ```bash
    python feature_engineering.py
    ```

## Model Training and Experiment Tracking

1. **Train Models and Track with MLflow:**

    ```bash
    python train_models.py
    python mlflow_tracking.py
    ```

## Model Deployment

1. **Build Docker Images:**

    ```bash
    docker build -t your-docker-repo/sklearn_model:latest -f Dockerfile .
    docker build -t your-docker-repo/statsmodels_model:latest -f Dockerfile .
    ```

2. **Push Docker Images:**

    ```bash
    docker push your-docker-repo/sklearn_model:latest
    docker push your-docker-repo/statsmodels_model:latest
    ```

3. **Deploy with Seldon Core:**

    ```bash
    kubectl apply -f seldon_deployment.yaml
    ```

## Running with Docker

To run the entire project using Docker, ensure that all services like MLflow and Feast are accessible from within the Docker container.

