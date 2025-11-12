# Archivo: eq18_turkish_music_mlops/utils/transformers.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def clean_finite_values(X):
    """
    Reemplaza NaN, inf y -inf de forma segura usando float32 
    (compatible con el pipeline de entrenamiento).
    """
    # np.nan_to_num SÍ arregla los NaN, inf y -inf
    return np.nan_to_num(
        X, 
        nan=0.0, 
        posinf=np.finfo(np.float32).max, 
        neginf=np.finfo(np.float32).min
    )

class OutlierIQRTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        X_clipped = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_clipped.values