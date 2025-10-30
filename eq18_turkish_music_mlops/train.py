import yaml
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import argparse
import json


class OutlierIQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.lower = Q1 - self.factor * (Q3 - Q1)
        self.upper = Q3 + self.factor * (Q3 - Q1)
        return self

    def transform(self, X):
        return np.where((X < self.lower) | (X > self.upper), np.nan, X)

def get_model_and_params(model_name):
    """Retorna el modelo y su grid de hiperparámetros correspondiente."""
    if model_name == "logistic":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=1,
        )
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"],
            "fit_intercept": [True, False],
        }

    elif model_name == "randomforest":
        model = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    elif model_name == "xgboost":
        model = XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            eval_metric="mlogloss",
            random_state=1,
            use_label_encoder=False,
        )
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        }

    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")

    return model, param_grid
	
def main(model_name):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    processed_dir = Path(params["paths"]["processed_dir"])
    models_dir = Path(params["paths"]["models_dir"])
    reports_dir = Path(params["paths"]["reports_dir"])
    random_state = params["data"]["random_state"]
    target_col = params["data"]["target_col"]
    iqr_factor = params["processing"]["iqr_factor"]
    pca_variance = params["processing"]["pca_variance"]
    imputer_strategy = params["processing"]["imputer_strategy"]
    n_splits = params["training"]["n_splits"]
    kfold_shuffle = params["training"]["kfold_shuffle"]
    grid_search_cv_scoring = params["training"]["grid_search_cv_scoring"]
    grid_search_cv_n_jobs = params["training"]["grid_search_cv_n_jobs"]

    print("Cargando datos...")
    df = pd.read_csv(processed_dir / "train.csv")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Pipeline numérico
    numeric_pipeline = Pipeline([
        ("outliers", OutlierIQRTransformer(factor=iqr_factor)),
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=random_state)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols)
    ])

    # --- Seleccionar modelo y grid ---
    print(f"Entrenando modelo: {model_name}")
    model, param_grid = get_model_and_params(model_name)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("modelo", model)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=kfold_shuffle, random_state=random_state)
    param_grid_prefixed = {f"modelo__{k}": v for k, v in param_grid.items()}
    grid = GridSearchCV(full_pipeline,  param_grid_prefixed, cv=cv, scoring=grid_search_cv_scoring, n_jobs=grid_search_cv_n_jobs)
    grid.fit(X, y_encoded)

    print(f"Mejor modelo: {grid.best_params_}")
    print(f"Accuracy promedio: {grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, models_dir / f"model_{model_name}.pkl")

    with open(reports_dir / f"train_results_{model_name}.json", "w") as f:
        json.dump({
            "model": model_name,
            "best_params": grid.best_params_,
            "cv_accuracy": grid.best_score_,
        }, f, indent=4)

    print("Modelo y resultados guardados correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de clasificación de emociones musicales")
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "randomforest", "xgboost"],
        required=True,
        help="Modelo a entrenar (logistic, randomforest, xgboost)",
    )
    args = parser.parse_args()
	
    main(args.model)