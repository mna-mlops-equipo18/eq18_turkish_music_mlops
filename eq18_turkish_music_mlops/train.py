"""
Módulo de entrenamiento de modelos para Turkish Music Emotion.

Implementa:
1. Pipeline completo de sklearn (preprocessing + modelo)
2. Grid Search con validación cruzada estratificada
3. Soporte para múltiples modelos (Logistic, RandomForest, XGBoost)
4. Guardado de modelos y resultados

Uso:
    python -m eq18_turkish_music_mlops.train --model logistic
    python -m eq18_turkish_music_mlops.train --model randomforest
    python -m eq18_turkish_music_mlops.train --model xgboost
"""

import yaml
import pandas as pd
import joblib
import logging
import sys
import time
import argparse
import json
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from eq18_turkish_music_mlops.utils.transformers import OutlierIQRTransformer
from eq18_turkish_music_mlops.utils.mlflow import start_training_run
from eq18_turkish_music_mlops.utils.logger import setup_logging
from sklearn.preprocessing import FunctionTransformer

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# ... (imports de sklearn, pandas, etc.) ...
from sklearn.linear_model import LogisticRegression
# <-- YA NO IMPORTAS XGBOOST NI RANDOM FOREST AQUÍ

# ... (más código) ...

def get_model(model_name, random_state):
    """
    Retorna la instancia del modelo base según el nombre.
    """
    
    if model_name == "logistic":
        # Logistic Regression ya está importado arriba, está bien
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        logger.info("Modelo Logistic Regression inicializado")

    elif model_name == "randomforest":
        # --- Importa Random Forest AQUÍ ---
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1  # Usar todos los cores disponibles
        )
        logger.info("Modelo Random Forest inicializado")

    elif model_name == "xgboost":
        # --- Importa XGBoost AQUÍ ---
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            eval_metric="mlogloss",
            random_state=random_state,
            use_label_encoder=False,
            n_jobs=-1
        )
        logger.info("Modelo XGBoost inicializado")
        
    else:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. "
            f"Opciones válidas: logistic, randomforest, xgboost"
        )

    return model


# Reemplaza NaN e inf 
def clean_finite_values(X):
    return np.nan_to_num(
        X, 
        nan=0.0, 
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min 
    )
    

def load_params(params_path="params.yaml"):
    """
    Carga parámetros desde archivo YAML.
    
    Args:
        params_path (str): Ruta al archivo de parámetros
        
    Returns:
        dict: Diccionario con parámetros
    """
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        logger.info(f"Parámetros cargados desde {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Archivo {params_path} no encontrado")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear YAML: {e}")
        raise


def validate_pca_components(n_components, max_features):
    """
    Valida que los componentes PCA sean razonables.
    
    Args:
        n_components (float or int): Componentes PCA (varianza o número)
        max_features (int): Número máximo de features disponibles
        
    Raises:
        ValueError: Si n_components es inválido
    """
    if isinstance(n_components, float):
        if not 0.0 < n_components < 1.0:
            raise ValueError(
                f"PCA variance debe estar entre 0 y 1, recibido: {n_components}"
            )
    elif isinstance(n_components, int):
        if n_components > max_features:
            raise ValueError(
                f"PCA components ({n_components}) excede features disponibles ({max_features})"
            )


def create_preprocessing_pipeline(numeric_cols, params, random_state):
    """
    Crea el pipeline de preprocesamiento.
    """
    iqr_factor = params["processing"]["iqr_factor"]
    pca_variance = params["processing"]["pca_variance"]
    imputer_strategy = params["processing"]["imputer_strategy"]
    power_transform_method = params["processing"]["power_transform_method"]
    
    logger.info(f"Configurando preprocessing pipeline:")
    logger.info(f"  - Outlier IQR factor: {iqr_factor}")
    logger.info(f"  - Power transform method: {power_transform_method}")
    logger.info(f"  - PCA variance: {pca_variance}")
    logger.info(f"  - Imputer strategy: {imputer_strategy}")
    
    # Validar PCA
    validate_pca_components(pca_variance, len(numeric_cols))

    # Pipeline numérico
    numeric_pipeline = Pipeline([
        ("outliers", OutlierIQRTransformer(factor=iqr_factor)),
        ("imputer", SimpleImputer(strategy=imputer_strategy)), # Arregla NaNs originales
        ("power", PowerTransformer(method=power_transform_method)), # <-- Crea 'inf'
        ("scaler", StandardScaler()),
        
        # --- ¡ESTA ES LA LÍNEA CORRECTA! ---
        # Arregla CUALQUIER inf o NaN creado por los pasos anteriores
        ("cleanup_finite", FunctionTransformer(clean_finite_values, validate=False)), 
        
        ("pca", PCA(n_components=pca_variance, random_state=random_state)),
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols)
    ])
    
    return preprocessor


def save_label_encoder(label_encoder, models_dir):
    """
    Guarda el label encoder para uso en inferencia.
    
    Args:
        label_encoder (LabelEncoder): Encoder entrenado
        models_dir (Path): Directorio de modelos
    """
    encoder_path = models_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"Label encoder guardado en {encoder_path}")
    logger.info(f"Clases: {label_encoder.classes_.tolist()}")


def perform_grid_search(pipeline, param_grid, X, y, cv_config, scoring, n_jobs):
    """
    Ejecuta Grid Search con validación cruzada.
    
    Args:
        pipeline (Pipeline): Pipeline completo
        param_grid (dict): Grid de hiperparámetros
        X (pd.DataFrame): Features
        y (np.array): Target codificado
        cv_config (StratifiedKFold): Configuración de CV
        scoring (str): Métrica de scoring
        n_jobs (int): Número de jobs paralelos
        
    Returns:
        GridSearchCV: Objeto fitted con resultados
    """
    logger.info("Iniciando Grid Search...")
    logger.info(f"  - Configuración CV: {cv_config.n_splits} folds, shuffle={cv_config.shuffle}")
    logger.info(f"  - Scoring: {scoring}")
    logger.info(f"  - Combinaciones a probar: {len(list(param_grid.values())[0]) if param_grid else 1}")
    
    start_time = time.time()
    
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_config,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    grid.fit(X, y)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Grid Search completado en {elapsed_time:.2f} segundos ({elapsed_time/60:.2f} min)")
    
    return grid


def log_best_results(grid, model_name):
    """
    Registra los mejores resultados del Grid Search.
    
    Args:
        grid (GridSearchCV): Objeto Grid Search fitted
        model_name (str): Nombre del modelo
    """
    logger.info("="*60)
    logger.info(f"RESULTADOS PARA {model_name.upper()}")
    logger.info("="*60)
    logger.info(f"Mejores hiperparámetros:\n{json.dumps(grid.best_params_, indent=2)}")
    logger.info(f"Mejor CV score: {grid.best_score_:.4f}")
    logger.info(f"Std CV score: {grid.cv_results_['std_test_score'][grid.best_index_]:.4f}")
    
    # Top 3 configuraciones
    results_df = pd.DataFrame(grid.cv_results_)
    top_configs = results_df.nsmallest(3, 'rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    logger.info(f"Top 3 configuraciones:\n{top_configs}")


def save_model_and_results(grid, model_name, models_dir, reports_dir):
    """
    Guarda el mejor modelo y los resultados del entrenamiento.
    
    Args:
        grid (GridSearchCV): Objeto Grid Search fitted
        model_name (str): Nombre del modelo
        models_dir (Path): Directorio de modelos
        reports_dir (Path): Directorio de reportes
    """
    # Guardar modelo
    model_path = models_dir / f"model_{model_name}.pkl"
    joblib.dump(grid.best_estimator_, model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # Preparar resultados detallados
    results = {
        "model": model_name,
        "best_params": grid.best_params_,
        "cv_score_mean": float(grid.best_score_),
        "cv_score_std": float(grid.cv_results_['std_test_score'][grid.best_index_]),
        "n_splits": grid.cv.n_splits,
        "scoring": grid.scoring,
        "total_fit_time": float(grid.cv_results_['mean_fit_time'].sum()),
    }
    
    # Agregar feature importance si aplica
    if model_name in ["randomforest", "xgboost"]:
        try:
            best_model = grid.best_estimator_.named_steps["modelo"]
            if hasattr(best_model, 'feature_importances_'):
                results["feature_importance_available"] = True
                # No guardamos el array completo por tamaño, solo la disponibilidad
            else:
                results["feature_importance_available"] = False
        except Exception as e:
            logger.warning(f"No se pudo extraer feature importance: {e}")
    
    # Guardar reporte
    report_path = reports_dir / f"train_results_{model_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Reporte guardado en {report_path}")


def main(model_name):
    """
    Función principal del pipeline de entrenamiento.
    
    Args:
        model_name (str): Nombre del modelo a entrenar
    """
    try:
        # Crear directorios necesarios
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"INICIANDO ENTRENAMIENTO: {model_name.upper()}")
        logger.info("="*60)
        
        # 1. Cargar parámetros
        params = load_params()
        
        processed_dir = Path(params["paths"]["processed_dir"])
        models_dir = Path(params["paths"]["models_dir"])
        reports_dir = Path(params["paths"]["reports_dir"])
        random_state = params["data"]["random_state"]
        target_col = params["data"]["target_col"]
        n_splits = params["training"]["n_splits"]
        kfold_shuffle = params["training"]["kfold_shuffle"]
        grid_search_cv_scoring = params["training"]["grid_search_cv_scoring"]
        grid_search_cv_n_jobs = params["training"]["grid_search_cv_n_jobs"]
        
        # Obtener grid de hiperparámetros
        param_grid = params["hyperparameters"][model_name]
        logger.info(f"Grid de hiperparámetros:\n{json.dumps(param_grid, indent=2)}")
        
        # 2. Crear directorios de salida
        models_dir.mkdir(exist_ok=True, parents=True)
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        # 3. Cargar datos
        train_path = processed_dir / "train.csv"
        logger.info(f"Cargando datos de entrenamiento desde {train_path}")
        df = pd.read_csv(train_path)
        logger.info(f"Dataset cargado: {df.shape}")
        
        # 4. Separar features y target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features: {X.shape[1]} columnas")
        logger.info(f"Target: {y.value_counts().to_dict()}")
        
        # 5. Codificar target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        save_label_encoder(label_encoder, models_dir)
        
        # 6. Identificar columnas numéricas
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        logger.info(f"Columnas numéricas: {len(numeric_cols)}")
        
        # 7. Crear preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numeric_cols, params, random_state)
        
        # 8. Crear modelo
        model = get_model(model_name, random_state)
        
        # 9. Pipeline completo
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("modelo", model)
        ])
        
        # 10. Configurar CV
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=kfold_shuffle,
            random_state=random_state
        )
        
        # 11. Preparar grid con prefijos
        param_grid_prefixed = {f"modelo__{k}": v for k, v in param_grid.items()}
        
        # 12. Grid Search
        grid = perform_grid_search(
            full_pipeline,
            param_grid_prefixed,
            X,
            y_encoded,
            cv,
            grid_search_cv_scoring,
            grid_search_cv_n_jobs
        )
        
        # 13. Registrar resultados
        log_best_results(grid, model_name)
        
        # 14. Guardar modelo y resultados
        save_model_and_results(grid, model_name, models_dir, reports_dir)
        
        start_training_run(
            model_name=model_name,
            params=params,
            grid=grid,
            models_dir=models_dir,
            reports_dir=reports_dir
        )

        logger.info("="*60)
        logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error crítico en train.py: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos de clasificación de emociones musicales"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "randomforest", "xgboost"],
        required=True,
        help="Modelo a entrenar: logistic, randomforest, xgboost",
    )
    args = parser.parse_args()
    
    main(args.model)