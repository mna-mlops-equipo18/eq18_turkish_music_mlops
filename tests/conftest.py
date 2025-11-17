"""
Fixtures compartidas para pruebas del proyecto Turkish Music Emotion.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib


@pytest.fixture
def sample_params():
    """Parámetros de prueba simplificados."""
    return {
        "paths": {
            "raw_data": "data/raw/test_data.csv",
            "processed_dir": "data/processed",
            "models_dir": "models",
            "reports_dir": "reports",
        },
        "data": {
            "target_col": "Class",
            "valid_classes": ["happy", "sad", "angry", "relax"],
            "test_size": 0.2,
            "random_state": 42,
        },
        "processing": {
            "iqr_factor": 1.5,
            "pca_variance": 0.90,
            "imputer_strategy": "median",
            "power_transform_method": "yeo-johnson",
        },
        "training": {
            "n_splits": 3,
            "kfold_shuffle": True,
            "grid_search_cv_scoring": "f1_macro",
            "grid_search_cv_n_jobs": 1,
        },
        "hyperparameters": {
            "logistic": {
                "C": [0.1, 1.0],
                "class_weight": [None, "balanced"],
                "fit_intercept": [True],
            }
        },
        "mlflow": {
            "experiment_name": "test_experiment"
        }
    }


@pytest.fixture
def sample_dataframe():
    """DataFrame de prueba con estructura válida."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    data = {
        f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)
    }
    data["Class"] = np.random.choice(["happy", "sad", "angry", "relax"], n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_outliers():
    """DataFrame con outliers extremos."""
    np.random.seed(42)
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 100],  # 100 es outlier
        "f2": [10, 20, 30, 40, 50, 60],
        "f3": [-5, -10, 0, 5, 10, 1000],  # 1000 es outlier
        "Class": ["happy", "sad", "angry", "relax", "happy", "sad"]
    })
    return df


@pytest.fixture
def sample_train_test_split(sample_dataframe):
    """Train/test split para pruebas."""
    from sklearn.model_selection import train_test_split
    
    X = sample_dataframe.drop(columns=["Class"])
    y = sample_dataframe["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_label_encoder():
    """LabelEncoder pre-entrenado."""
    encoder = LabelEncoder()
    encoder.fit(["happy", "sad", "angry", "relax"])
    return encoder


@pytest.fixture
def temp_workspace(tmp_path, sample_params):
    """Workspace temporal con estructura de directorios."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    
    # Crear directorios
    (workspace / "data/raw").mkdir(parents=True)
    (workspace / "data/processed").mkdir(parents=True)
    (workspace / "models").mkdir(parents=True)
    (workspace / "reports").mkdir(parents=True)
    (workspace / "logs").mkdir(parents=True)
    
    # Guardar params.yaml
    params_path = workspace / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(sample_params, f)
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(workspace)


@pytest.fixture
def sample_raw_csv(temp_workspace, sample_dataframe):
    """CSV raw de prueba."""
    csv_path = temp_workspace / "data/raw/test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_trained_model(sample_train_test_split, sample_label_encoder):
    """Modelo simple pre-entrenado para pruebas."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X_train, _, y_train, _ = sample_train_test_split
    y_train_encoded = sample_label_encoder.transform(y_train)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    model.fit(X_train, y_train_encoded)
    
    return model


@pytest.fixture
def mock_mlflow_run():
    """Mock de MLflow run para pruebas."""
    class MockRun:
        class Info:
            run_id = "test_run_12345"
        
        info = Info()
    
    return MockRun()
