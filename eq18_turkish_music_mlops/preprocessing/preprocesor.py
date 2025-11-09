import logging
import numpy as np
import pandas as pd
from sklearn import set_config
from typing import List, Optional
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder

class DataPreprocessor:

    def __init__(self, numeric_cols:Optional[List[str]]=None, pca_n_components:float=0.90):
        self.numeric_cols = numeric_cols
        self.pca_n_components = pca_n_components
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder = LabelEncoder()


    def _build_tabular(self, X:pd.DataFrame):
        try:
            num_cols = self.numeric_cols or X.select_dtypes(include=["number"]).columns.to_list()

            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scaler", StandardScaler())
                ("pca", PCA(n_components=self.pca_n_components))
            ] 

            numeric_pipeline = Pipeline(steps)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, num_cols)
                ],
                remainder='passthrough'
            )

            set_config(transform_output="pandas")
            preprocessor.set_output(transform="pandas")

            return preprocessor
        except Exception as e:
            logging.error(f"[Preprocesor] Error building tabular preprocessor")
            raise


    def _fit_transform_tabular(self, X:pd.DataFrame):
        try:
            self.preprocessor = self._build_tabular(X)
            return self.preprocessor.fit_transform(X)
        
        except Exception as e:
            logging.error(f"[Preprocesor] Error in fit transform dataset")
            raise


    def _fit_transform_target(self, y:pd.Series):
        try:
            return self.label_encoder.fit_transform(y)
        except Exception as e:
            logging.error(f"[Preprocesor] Error building target dataset")
            raise