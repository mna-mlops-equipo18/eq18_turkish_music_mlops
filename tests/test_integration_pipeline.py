"""
Pruebas de integración end-to-end para el pipeline completo.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import joblib
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.prepare import main as prepare_main
from eq18_turkish_music_mlops.train import main as train_main
from eq18_turkish_music_mlops.evaluate import main as evaluate_main


class TestPipelineIntegration:
    """Pruebas de integración del pipeline completo."""
    
    @pytest.fixture
    def setup_complete_workspace(self, temp_workspace, sample_dataframe):
        """Prepara workspace completo para pruebas E2E."""
        # Guardar CSV raw
        raw_path = temp_workspace / "data/raw/turkish_music_emotion_modified.csv"
        sample_dataframe.to_csv(raw_path, index=False)
        
        # Actualizar params.yaml con paths correctos
        params_path = temp_workspace / "params.yaml"
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        
        params["paths"]["raw_data"] = str(raw_path)
        
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        
        return temp_workspace
    
    def test_prepare_creates_outputs(self, setup_complete_workspace, monkeypatch):
        """Prepare crea train.csv y test.csv."""
        workspace = setup_complete_workspace
        monkeypatch.chdir(workspace)
        
        # Ejecutar prepare
        prepare_main()
        
        # Verificar outputs
        train_path = workspace / "data/processed/train.csv"
        test_path = workspace / "data/processed/test.csv"
        
        assert train_path.exists()
        assert test_path.exists()
        
        # Verificar contenido
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert "Class" in train_df.columns
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_train_creates_model(self, mock_mlflow, setup_complete_workspace, monkeypatch):
        """Train crea modelo y encoder."""
        workspace = setup_complete_workspace
        monkeypatch.chdir(workspace)
        
        # Mock MLflow
        mock_run = type('MockRun', (), {'info': type('Info', (), {'run_id': 'test_123'})})()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Ejecutar prepare primero
        prepare_main()
        
        # Modificar params para grid pequeño
        params_path = workspace / "params.yaml"
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        
        params["hyperparameters"]["logistic"]["C"] = [1.0]
        params["hyperparameters"]["logistic"]["class_weight"] = [None]
        params["hyperparameters"]["logistic"]["fit_intercept"] = [True]
        params["training"]["n_splits"] = 2
        
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        
        # Ejecutar train
        train_main("logistic")
        
        # Verificar outputs
        model_path = workspace / "models/model_logistic.pkl"
        encoder_path = workspace / "models/label_encoder.pkl"
        report_path = workspace / "reports/train_results_logistic.json"
        
        assert model_path.exists()
        assert encoder_path.exists()
        assert report_path.exists()
        
        # Verificar que se puede cargar el modelo
        model = joblib.load(model_path)
        assert hasattr(model, 'predict')
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_evaluate_generates_metrics(self, mock_mlflow, setup_complete_workspace, monkeypatch):
        """Evaluate genera métricas correctamente."""
        workspace = setup_complete_workspace
        monkeypatch.chdir(workspace)
        
        # Mock MLflow
        mock_run = type('MockRun', (), {'info': type('Info', (), {'run_id': 'eval_123'})})()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Ejecutar pipeline completo
        prepare_main()
        
        # Simplificar grid
        params_path = workspace / "params.yaml"
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        params["hyperparameters"]["logistic"]["C"] = [1.0]
        params["hyperparameters"]["logistic"]["class_weight"] = [None]
        params["hyperparameters"]["logistic"]["fit_intercept"] = [True]
        params["training"]["n_splits"] = 2
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        
        train_main("logistic")
        evaluate_main("logistic")
        
        # Verificar reporte de evaluación
        eval_report_path = workspace / "reports/evaluate_results_logistic.json"
        assert eval_report_path.exists()
        
        # Verificar contenido del reporte
        import json
        with open(eval_report_path, "r") as f:
            results = json.load(f)
        
        assert "metrics" in results
        assert "accuracy" in results["metrics"]
        assert "f1_macro" in results["metrics"]
        assert 0 <= results["metrics"]["accuracy"] <= 1
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_full_pipeline_end_to_end(self, mock_mlflow, setup_complete_workspace, monkeypatch):
        """Pipeline completo: prepare → train → evaluate."""
        workspace = setup_complete_workspace
        monkeypatch.chdir(workspace)
        
        # Mock MLflow
        mock_run = type('MockRun', (), {'info': type('Info', (), {'run_id': 'e2e_123'})})()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Simplificar params
        params_path = workspace / "params.yaml"
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        params["hyperparameters"]["logistic"]["C"] = [1.0]
        params["hyperparameters"]["logistic"]["class_weight"] = [None]
        params["hyperparameters"]["logistic"]["fit_intercept"] = [True]
        params["training"]["n_splits"] = 2
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        
        # Ejecutar pipeline
        prepare_main()
        train_main("logistic")
        evaluate_main("logistic")
        
        # Verificar todos los outputs críticos existen
        critical_files = [
            "data/processed/train.csv",
            "data/processed/test.csv",
            "models/model_logistic.pkl",
            "models/label_encoder.pkl",
            "reports/train_results_logistic.json",
            "reports/evaluate_results_logistic.json",
        ]
        
        for file in critical_files:
            assert (workspace / file).exists(), f"Missing: {file}"
    
    def test_predictions_are_consistent(self, setup_complete_workspace, monkeypatch):
        """Predicciones son consistentes con mismo modelo."""
        workspace = setup_complete_workspace
        monkeypatch.chdir(workspace)
        
        # Ejecutar pipeline mínimo
        prepare_main()
        
        # Cargar datos de test
        test_df = pd.read_csv(workspace / "data/processed/test.csv")
        X_test = test_df.drop(columns=["Class"])
        
        # Mock simple de modelo (sin entrenar real por tiempo)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.pipeline import Pipeline
        
        y_train = pd.read_csv(workspace / "data/processed/train.csv")["Class"]
        encoder = LabelEncoder()
        encoder.fit(y_train)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=100))
        ])
        
        train_df = pd.read_csv(workspace / "data/processed/train.csv")
        X_train = train_df.drop(columns=["Class"])
        y_train_enc = encoder.transform(train_df["Class"])
        
        model.fit(X_train, y_train_enc)
        
        # Predicciones múltiples deben ser idénticas
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        np.testing.assert_array_equal(pred1, pred2)
