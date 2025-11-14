"""
Transformadores personalizados para el pipeline de Machine Learning.

Este módulo contiene transformadores sklearn-compatible que se integran
en los pipelines de preprocesamiento para detección y manejo de outliers.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_finite_values(data):
    """
    Clip valores infinitos a rangos finitos.
    """
    return np.clip(data, -1e10, 1e10)


class OutlierIQRTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que detecta y marca outliers usando el método IQR.
    
    Este transformador identifica valores atípicos basándose en el rango
    intercuartílico (IQR) y los reemplaza con NaN para ser manejados
    posteriormente por un imputer.
    
    La regla de detección es:
        outlier si: x < Q1 - factor*(Q3-Q1) o x > Q3 + factor*(Q3-Q1)
    
    Donde:
        - Q1: Primer cuartil (percentil 25)
        - Q3: Tercer cuartil (percentil 75)
        - IQR = Q3 - Q1
        - factor: Multiplicador del IQR (típicamente 1.5)
    
    Attributes:
        factor (float): Multiplicador del IQR para definir límites de outliers.
                       Valores típicos: 1.5 (estándar), 3.0 (extremos)
        lower (np.array): Límites inferiores por feature (fit)
        upper (np.array): Límites superiores por feature (fit)
    
    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.impute import SimpleImputer
        >>> 
        >>> pipeline = Pipeline([
        ...     ('outliers', OutlierIQRTransformer(factor=1.5)),
        ...     ('imputer', SimpleImputer(strategy='median'))
        ... ])
        >>> 
        >>> X_clean = pipeline.fit_transform(X_train)
    
    Notes:
        - Los outliers se marcan como NaN, no se eliminan
        - Los límites se calculan solo en el conjunto de entrenamiento (fit)
        - En transform, se aplican los mismos límites a nuevos datos
        - Compatible con sklearn pipelines y ColumnTransformer
        - Funciona a nivel de columna (cada feature independiente)
    """
    
    def __init__(self, factor=1.5):
        """
        Inicializa el transformador de outliers.
        
        Args:
            factor (float, optional): Multiplicador del IQR para límites.
                                     Default: 1.5 (definición estándar de outliers)
                                     Aumentar a 2.0 o 3.0 para ser más permisivo.
        
        Raises:
            ValueError: Si factor <= 0
        """
        if factor <= 0:
            raise ValueError(f"factor debe ser positivo, recibido: {factor}")
        
        self.factor = factor
        self.lower = None
        self.upper = None
    
    def fit(self, X, y=None):
        """
        Calcula los límites de outliers basados en IQR del dataset de entrenamiento.
        
        Args:
            X (array-like, shape (n_samples, n_features)): Datos de entrenamiento
            y (array-like, optional): Target (ignorado, incluido por compatibilidad sklearn)
        
        Returns:
            self: Instancia fitted del transformador
        
        Notes:
            - Calcula Q1, Q3 e IQR para cada columna
            - Define límites: lower = Q1 - factor*IQR, upper = Q3 + factor*IQR
            - Los límites se almacenan para uso en transform()
        """
        X = np.asarray(X)
        
        # Calcular cuartiles por columna
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        # Calcular límites
        self.lower = Q1 - self.factor * IQR
        self.upper = Q3 + self.factor * IQR
        
        # Log información útil
        n_features = X.shape[1]
        logger.debug(f"OutlierIQRTransformer fitted con factor={self.factor}")
        logger.debug(f"  Features: {n_features}")
        logger.debug(f"  Rango IQR medio: {np.mean(IQR):.3f}")
        
        return self
    
    def transform(self, X):
        """
        Marca outliers como NaN usando los límites calculados en fit.
        
        Args:
            X (array-like, shape (n_samples, n_features)): Datos a transformar
        
        Returns:
            np.array: Datos con outliers reemplazados por NaN
        
        Raises:
            ValueError: Si transform() se llama antes de fit()
        
        Notes:
            - Usa los límites calculados en fit()
            - Outliers se marcan como NaN para ser imputados después
            - No modifica el shape del array
        """
        if self.lower is None or self.upper is None:
            raise ValueError("Transformador no ha sido fitted. Llama fit() primero.")
        
        X = np.asarray(X)
        
        # Contar outliers antes de transformar
        outliers_mask = (X < self.lower) | (X > self.upper)
        n_outliers = np.sum(outliers_mask)
        outlier_pct = (n_outliers / X.size) * 100
        
        if n_outliers > 0:
            logger.debug(f"Outliers detectados: {n_outliers}/{X.size} ({outlier_pct:.2f}%)")
        
        # Marcar outliers como NaN
        X_transformed = np.where(outliers_mask, np.nan, X)
        
        return X_transformed
    
    def get_outlier_stats(self, X):
        """
        Retorna estadísticas de outliers sin modificar los datos.
        
        Método auxiliar para análisis exploratorio.
        
        Args:
            X (array-like): Datos a analizar
        
        Returns:
            dict: Diccionario con estadísticas por feature
        
        Example:
            >>> outlier_detector = OutlierIQRTransformer(factor=1.5)
            >>> outlier_detector.fit(X_train)
            >>> stats = outlier_detector.get_outlier_stats(X_train)
            >>> print(f"Outliers en feature 0: {stats['outlier_counts'][0]}")
        """
        if self.lower is None or self.upper is None:
            raise ValueError("Transformador no ha sido fitted.")
        
        X = np.asarray(X)
        outliers_mask = (X < self.lower) | (X > self.upper)
        
        stats = {
            "outlier_counts": np.sum(outliers_mask, axis=0).tolist(),
            "outlier_percentages": (np.sum(outliers_mask, axis=0) / X.shape[0] * 100).tolist(),
            "lower_bounds": self.lower.tolist(),
            "upper_bounds": self.upper.tolist()
        }
        
        return stats