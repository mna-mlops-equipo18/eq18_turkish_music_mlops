"""
Pruebas unitarias para eq18_turkish_music_mlops/utils/transformers.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.utils.transformers import (
    OutlierIQRTransformer,
    clean_finite_values
)


class TestCleanFiniteValues:
    """Pruebas para función clean_finite_values."""
    
    def test_clip_infinite_values(self):
        """Valores infinitos se recortan a rango finito."""
        X = np.array([[1, 2, np.inf], [3, -np.inf, 5]])
        X_clean = clean_finite_values(X)
        
        assert np.all(np.isfinite(X_clean))
        assert X_clean[0, 2] == 1e6  # inf → 1e6
        assert X_clean[1, 1] == -1e6  # -inf → -1e6
    
    def test_preserve_finite_values(self):
        """Valores finitos no cambian."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_clean = clean_finite_values(X)
        
        np.testing.assert_array_equal(X, X_clean)


class TestOutlierIQRTransformer:
    """Pruebas para OutlierIQRTransformer."""
    
    def test_initialization_valid_factor(self):
        """Inicialización correcta con factor válido."""
        transformer = OutlierIQRTransformer(factor=1.5)
        assert transformer.factor == 1.5
        assert transformer.lower is None
        assert transformer.upper is None
    
    def test_initialization_invalid_factor(self):
        """Error con factor inválido."""
        with pytest.raises(ValueError, match="positivo"):
            OutlierIQRTransformer(factor=-1)
        
        with pytest.raises(ValueError, match="positivo"):
            OutlierIQRTransformer(factor=0)
    
    def test_fit_calculates_bounds(self):
        """Fit calcula límites correctamente."""
        X = np.array([[1], [2], [3], [4], [5], [100]])  # 100 es outlier
        
        transformer = OutlierIQRTransformer(factor=1.5)
        transformer.fit(X)
        
        # Q1=2, Q3=5, IQR=3
        # lower = 2 - 1.5*3 = -2.5
        # upper = 5 + 1.5*3 = 9.5
        
        assert transformer.lower is not None
        assert transformer.upper is not None
        assert transformer.lower[0] < 1
        assert transformer.upper[0] < 100
    
    def test_transform_marks_outliers_as_nan(self):
        """Transform marca outliers como NaN."""
        X = np.array([[1], [2], [3], [4], [5], [100]])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        # 100 debe ser NaN
        assert np.isnan(X_transformed[-1, 0])
        
        # Valores normales preservados
        assert X_transformed[0, 0] == 1
        assert X_transformed[2, 0] == 3
    
    def test_transform_without_fit_raises_error(self):
        """Error si transform se llama antes de fit."""
        transformer = OutlierIQRTransformer(factor=1.5)
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="no ha sido fitted"):
            transformer.transform(X)
    
    def test_fit_transform(self):
        """fit_transform funciona correctamente."""
        X = np.array([[1], [2], [3], [4], [5], [100]])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        X_transformed = transformer.fit_transform(X)
        
        assert np.isnan(X_transformed[-1, 0])
        assert not np.isnan(X_transformed[0, 0])
    
    def test_multivariate_data(self):
        """Funciona con múltiples features."""
        X = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50],
            [100, 1000]  # Outliers en ambas columnas
        ])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        X_transformed = transformer.fit_transform(X)
        
        # Última fila debe tener NaNs
        assert np.isnan(X_transformed[-1, 0])
        assert np.isnan(X_transformed[-1, 1])
        
        # Primera fila normal
        assert not np.isnan(X_transformed[0, 0])
        assert not np.isnan(X_transformed[0, 1])
    
    def test_different_factors(self):
        """Factor más alto detecta menos outliers."""
        X = np.array([[1], [2], [3], [4], [5], [10], [100]])
        
        # Factor bajo = más outliers
        transformer_strict = OutlierIQRTransformer(factor=1.5)
        X_strict = transformer_strict.fit_transform(X)
        
        # Factor alto = menos outliers
        transformer_loose = OutlierIQRTransformer(factor=3.0)
        X_loose = transformer_loose.fit_transform(X)
        
        # Contar NaNs
        nan_count_strict = np.sum(np.isnan(X_strict))
        nan_count_loose = np.sum(np.isnan(X_loose))
        
        assert nan_count_strict >= nan_count_loose
    
    def test_get_outlier_stats(self):
        """get_outlier_stats retorna estadísticas correctas."""
        X = np.array([[1], [2], [3], [4], [5], [100]])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        transformer.fit(X)
        
        stats = transformer.get_outlier_stats(X)
        
        assert "outlier_counts" in stats
        assert "outlier_percentages" in stats
        assert "lower_bounds" in stats
        assert "upper_bounds" in stats
        
        # 1 outlier en 6 muestras ≈ 16.67%
        assert stats["outlier_counts"][0] == 1
        assert 15 < stats["outlier_percentages"][0] < 20
    
    def test_get_outlier_stats_without_fit(self):
        """Error al llamar get_outlier_stats sin fit."""
        transformer = OutlierIQRTransformer(factor=1.5)
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="no ha sido fitted"):
            transformer.get_outlier_stats(X)
    
    def test_no_outliers_case(self):
        """Sin outliers, no marca ningún NaN."""
        X = np.array([[1], [2], [3], [4], [5]])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        X_transformed = transformer.fit_transform(X)
        
        # Ningún valor debe ser NaN
        assert not np.any(np.isnan(X_transformed))
    
    def test_all_same_values(self):
        """Datos con misma valor (IQR=0)."""
        X = np.array([[5], [5], [5], [5], [5]])
        
        transformer = OutlierIQRTransformer(factor=1.5)
        X_transformed = transformer.fit_transform(X)
        
        # No debe haber NaNs
        assert not np.any(np.isnan(X_transformed))
