"""
Pruebas unitarias para eq18_turkish_music_mlops/evaluate.py
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.evaluate import (
    load_test_data,
    validate_consistency,
    make_predictions,
    calculate_metrics,
)


class TestLoadTestData:
    """Pruebas para carga de datos de test."""
    
    def test_load_valid_csv(self, temp_workspace, sample_dataframe):
        """Carga exitosa de CSV válido."""
        test_path = temp_workspace / "data/processed/test.csv"
        sample_dataframe.to_csv(test_path, index=False)
        
        X_test, y_true = load_test_data(test_path, "Class")
        
        assert len(X_test) == len(sample_dataframe)
        assert "Class" not in X_test.columns
        assert len(y_true) == len(sample_dataframe)
    
    def test_load_missing_file(self, temp_workspace):
        """Error al cargar archivo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_test_data(temp_workspace / "nonexistent.csv", "Class")


class TestValidateConsistency:
    """Pruebas para validación de consistencia de clases."""
    
    def test_consistent_classes(self, sample_label_encoder):
        """No error cuando clases coinciden."""
        y_true = pd.Series(["happy", "sad", "angry", "relax"])
        
        # No debe lanzar excepción
        validate_consistency(y_true, sample_label_encoder)
    
    def test_missing_class_in_test(self, sample_label_encoder):
        """Warning cuando clase de train falta en test."""
        y_true = pd.Series(["happy", "sad", "angry"])  # falta relax
        
        # Debe completar sin error (solo warning)
        validate_consistency(y_true, sample_label_encoder)
    
    def test_extra_class_in_test(self, sample_label_encoder):
        """Error cuando test tiene clase no vista en train."""
        y_true = pd.Series(["happy", "sad", "angry", "unknown"])
        
        with pytest.raises(ValueError, match="no en train"):
            validate_consistency(y_true, sample_label_encoder)


class TestMakePredictions:
    """Pruebas para generación de predicciones."""
    
    def test_predictions_shape(self, sample_trained_model, sample_train_test_split, sample_label_encoder):
        """Predicciones tienen shape correcto."""
        _, X_test, _, _ = sample_train_test_split
        
        y_pred_encoded, y_pred = make_predictions(
            sample_trained_model, X_test, sample_label_encoder
        )
        
        assert len(y_pred_encoded) == len(X_test)
        assert len(y_pred) == len(X_test)
    
    def test_predictions_valid_classes(self, sample_trained_model, sample_train_test_split, sample_label_encoder):
        """Predicciones son clases válidas."""
        _, X_test, _, _ = sample_train_test_split
        
        _, y_pred = make_predictions(
            sample_trained_model, X_test, sample_label_encoder
        )
        
        valid_classes = set(sample_label_encoder.classes_)
        assert all(pred in valid_classes for pred in y_pred)


class TestCalculateMetrics:
    """Pruebas para cálculo de métricas."""
    
    def test_perfect_predictions(self, sample_label_encoder):
        """Métricas con predicciones perfectas."""
        y_true = pd.Series(["happy", "sad", "angry", "relax"])
        y_pred = np.array(["happy", "sad", "angry", "relax"])
        
        y_true_encoded = sample_label_encoder.transform(y_true)
        y_pred_encoded = sample_label_encoder.transform(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
    
    def test_worst_predictions(self, sample_label_encoder):
        """Métricas con predicciones completamente erróneas."""
        y_true = pd.Series(["happy", "sad", "angry", "relax"])
        y_pred = np.array(["sad", "angry", "relax", "happy"])
        
        y_true_encoded = sample_label_encoder.transform(y_true)
        y_pred_encoded = sample_label_encoder.transform(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded)
        
        assert metrics["accuracy"] == 0.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
    
    def test_metrics_structure(self, sample_label_encoder):
        """Estructura de métricas es correcta."""
        y_true = pd.Series(["happy", "sad", "angry", "relax"])
        y_pred = np.array(["happy", "sad", "angry", "relax"])
        
        y_true_encoded = sample_label_encoder.transform(y_true)
        y_pred_encoded = sample_label_encoder.transform(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
    
    def test_confusion_matrix_shape(self, sample_label_encoder):
        """Matriz de confusión tiene dimensiones correctas."""
        y_true = pd.Series(["happy", "sad", "angry", "relax"] * 5)
        y_pred = np.array(["happy", "sad", "angry", "relax"] * 5)
        
        y_true_encoded = sample_label_encoder.transform(y_true)
        y_pred_encoded = sample_label_encoder.transform(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded)
        
        cm = metrics["confusion_matrix"]
        n_classes = len(sample_label_encoder.classes_)
        
        assert len(cm) == n_classes
        assert len(cm[0]) == n_classes
