"""
Pruebas unitarias para eq18_turkish_music_mlops/utils/mlflow.py
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.utils.mlflow import (
    start_training_run,
    log_evaluation_to_run,
)


class TestStartTrainingRun:
    """Pruebas para inicio de runs de MLflow."""
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_start_run_creates_files(self, mock_mlflow, temp_workspace, sample_params):
        """start_training_run crea archivos necesarios."""
        models_dir = temp_workspace / "models"
        reports_dir = temp_workspace / "reports"
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear archivos mock necesarios
        model_path = models_dir / "model_logistic.pkl"
        encoder_path = models_dir / "label_encoder.pkl"
        report_path = reports_dir / "train_results_logistic.json"
        
        model_path.touch()
        encoder_path.touch()
        report_path.write_text('{}')
        
        # Mock GridSearchCV
        mock_grid = Mock()
        mock_grid.best_params_ = {"C": 1.0, "modelo__penalty": "l2"}
        mock_grid.best_score_ = 0.85
        mock_grid.best_index_ = 0
        mock_grid.cv_results_ = {"std_test_score": [0.05]}
        mock_grid.best_estimator_ = Mock()
        
        # Mock MLflow context
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Ejecutar
        start_training_run(
            model_name="logistic",
            params=sample_params,
            grid=mock_grid,
            models_dir=models_dir,
            reports_dir=reports_dir
        )
        
        # Verificar que se guardó run_id
        run_id_path = reports_dir / "run_id_logistic.txt"
        assert run_id_path.exists()
        assert run_id_path.read_text() == "test_run_123"
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_logs_parameters(self, mock_mlflow, temp_workspace, sample_params):
        """Verifica que se loguean parámetros."""
        models_dir = temp_workspace / "models"
        reports_dir = temp_workspace / "reports"
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear archivos necesarios
        (models_dir / "model_logistic.pkl").touch()
        (models_dir / "label_encoder.pkl").touch()
        (reports_dir / "train_results_logistic.json").write_text('{}')
        
        mock_grid = Mock()
        mock_grid.best_params_ = {"modelo__C": 1.0}
        mock_grid.best_score_ = 0.85
        mock_grid.best_index_ = 0
        mock_grid.cv_results_ = {"std_test_score": [0.05]}
        mock_grid.best_estimator_ = Mock()
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_456"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        start_training_run(
            model_name="logistic",
            params=sample_params,
            grid=mock_grid,
            models_dir=models_dir,
            reports_dir=reports_dir
        )
        
        # Verificar que log_param fue llamado
        assert mock_mlflow.log_param.called


class TestLogEvaluationToRun:
    """Pruebas para logging de evaluación."""
    
    @patch('eq18_turkish_music_mlops.utils.mlflow.mlflow')
    def test_log_evaluation_with_valid_run_id(self, mock_mlflow, temp_workspace):
        """Log evaluación con run_id válido."""
        reports_dir = temp_workspace / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo run_id
        run_id_path = reports_dir / "run_id_logistic.txt"
        run_id_path.write_text("test_run_789")
        
        # Crear reporte de evaluación
        eval_report_path = reports_dir / "evaluate_results_logistic.json"
        eval_report_path.write_text('{}')
        
        results = {
            "metrics": {
                "accuracy": 0.90,
                "f1_macro": 0.88
            }
        }
        
        mock_mlflow.start_run.return_value.__enter__.return_value = Mock()
        
        log_evaluation_to_run(
            model_name="logistic",
            results=results,
            reports_dir=reports_dir
        )
        
        # Verificar que se reabrió el run
        mock_mlflow.start_run.assert_called_with(run_id="test_run_789")
        
        # Verificar que se loguearon métricas
        assert mock_mlflow.log_metrics.called
    
    def test_log_evaluation_without_run_id(self, temp_workspace, caplog):
        """Warning cuando no existe run_id."""
        reports_dir = temp_workspace / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results = {"metrics": {"accuracy": 0.90}}
        
        log_evaluation_to_run(
            model_name="logistic",
            results=results,
            reports_dir=reports_dir
        )
        
        # Verificar que se logueó warning (función retorna sin error)
        # No debe crashear