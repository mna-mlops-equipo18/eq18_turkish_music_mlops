"""
Pruebas unitarias para eq18_turkish_music_mlops/train.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.train import (
    get_model,
    validate_pca_components,
    create_preprocessing_pipeline,
)


class TestGetModel:
    """Pruebas para inicialización de modelos."""
    
    def test_logistic_regression(self):
        """Logistic Regression se inicializa correctamente."""
        model = get_model("logistic", random_state=42)
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_random_forest(self):
        """Random Forest se inicializa correctamente."""
        model = get_model("randomforest", random_state=42)
        
        assert model is not None
        assert hasattr(model, 'fit')
    
    def test_xgboost(self):
        """XGBoost se inicializa correctamente."""
        model = get_model("xgboost", random_state=42)
        
        assert model is not None
        assert hasattr(model, 'fit')
    
    def test_invalid_model_name(self):
        """Error con nombre de modelo inválido."""
        with pytest.raises(ValueError, match="no soportado"):
            get_model("invalid_model", random_state=42)


class TestValidatePCAComponents:
    """Pruebas para validación de componentes PCA."""
    
    def test_valid_variance_float(self):
        """Varianza float válida (0 < x < 1)."""
        # No debe lanzar excepción
        validate_pca_components(0.90, max_features=10)
        validate_pca_components(0.5, max_features=10)
    
    def test_invalid_variance_float(self):
        """Error con varianza fuera de rango."""
        with pytest.raises(ValueError, match="entre 0 y 1"):
            validate_pca_components(1.5, max_features=10)
        
        with pytest.raises(ValueError, match="entre 0 y 1"):
            validate_pca_components(-0.1, max_features=10)
    
    def test_valid_components_int(self):
        """Número de componentes válido."""
        validate_pca_components(5, max_features=10)
    
    def test_invalid_components_int(self):
        """Error cuando componentes > features."""
        with pytest.raises(ValueError, match="excede features"):
            validate_pca_components(15, max_features=10)


class TestCreatePreprocessingPipeline:
    """Pruebas para creación del pipeline de preprocesamiento."""
    
    def test_pipeline_structure(self, sample_params):
        """Pipeline tiene estructura correcta."""
        numeric_cols = ["f1", "f2", "f3"]
        
        pipeline = create_preprocessing_pipeline(
            numeric_cols, sample_params, random_state=42
        )
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    def test_pipeline_fit_transform(self, sample_params, sample_dataframe):
        """Pipeline puede fit y transform."""
        X = sample_dataframe.drop(columns=["Class"])
        numeric_cols = X.columns.tolist()
        
        pipeline = create_preprocessing_pipeline(
            numeric_cols, sample_params, random_state=42
        )
        
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        # Dimensión reducida por PCA
        assert X_transformed.shape[1] <= X.shape[1]
    
    def test_pipeline_components_exist(self, sample_params):
        """Pipeline contiene todos los componentes esperados."""
        numeric_cols = ["f1", "f2", "f3"]
        
        pipeline = create_preprocessing_pipeline(
            numeric_cols, sample_params, random_state=42
        )
        
        # Verificar que es un ColumnTransformer
        from sklearn.compose import ColumnTransformer
        assert isinstance(pipeline, ColumnTransformer)
