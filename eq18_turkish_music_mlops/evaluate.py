"""
Módulo de evaluación de modelos para Turkish Music Emotion.

Realiza:
1. Carga de modelo entrenado y datos de test
2. Generación de predicciones
3. Cálculo de métricas (accuracy, F1, precision, recall)
4. Matriz de confusión y reporte de clasificación
5. Guardado de resultados en JSON

Uso:
    python -m eq18_turkish_music_mlops.evaluate --model logistic
    python -m eq18_turkish_music_mlops.evaluate --model randomforest
    python -m eq18_turkish_music_mlops.evaluate --model xgboost
"""

import json
import yaml
import joblib
import pandas as pd
import logging
import sys
import time
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import argparse

from eq18_turkish_music_mlops.utils.mlflow import log_evaluation_to_run

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


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


def load_test_data(test_path, target_col):
    """
    Carga datos de test y separa features/target.
    
    Args:
        test_path (Path): Ruta al archivo test.csv
        target_col (str): Nombre de columna objetivo
        
    Returns:
        tuple: (X_test, y_true)
    """
    logger.info(f"Cargando datos de test desde {test_path}")
    df_test = pd.read_csv(test_path)
    logger.info(f"Dataset de test: {df_test.shape}")
    
    X_test = df_test.drop(columns=[target_col])
    y_true = df_test[target_col]
    
    logger.info(f"Features: {X_test.shape[1]}, Samples: {len(y_true)}")
    logger.info(f"Distribución de clases:\n{y_true.value_counts()}")
    
    return X_test, y_true


def load_model_and_encoder(models_dir, model_name):
    """
    Carga modelo entrenado y label encoder.
    
    Args:
        models_dir (Path): Directorio de modelos
        model_name (str): Nombre del modelo
        
    Returns:
        tuple: (model, label_encoder)
        
    Raises:
        FileNotFoundError: Si no existen los archivos
    """
    model_path = models_dir / f"model_{model_name}.pkl"
    encoder_path = models_dir / "label_encoder.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder no encontrado: {encoder_path}")
    
    logger.info(f"Cargando modelo desde {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Cargando label encoder desde {encoder_path}")
    label_encoder = joblib.load(encoder_path)
    logger.info(f"Clases del encoder: {label_encoder.classes_.tolist()}")
    
    return model, label_encoder


def validate_consistency(y_true, label_encoder):
    """
    Valida que las clases en test coincidan con las del encoder.
    
    Args:
        y_true (pd.Series): Target real
        label_encoder (LabelEncoder): Encoder entrenado
        
    Raises:
        ValueError: Si hay inconsistencias
    """
    test_classes = set(y_true.unique())
    train_classes = set(label_encoder.classes_)
    
    if test_classes != train_classes:
        missing_in_test = train_classes - test_classes
        extra_in_test = test_classes - train_classes
        
        if missing_in_test:
            logger.warning(f"Clases en train pero no en test: {missing_in_test}")
        if extra_in_test:
            raise ValueError(f"Clases en test pero no en train: {extra_in_test}")
    else:
        logger.info("Consistencia de clases validada correctamente")


def make_predictions(model, X_test, label_encoder):
    """
    Genera predicciones sobre datos de test.
    
    Args:
        model: Modelo entrenado
        X_test (pd.DataFrame): Features de test
        label_encoder (LabelEncoder): Encoder para decodificar
        
    Returns:
        tuple: (y_pred_encoded, y_pred)
    """
    logger.info("Generando predicciones...")
    
    start_time = time.time()
    y_pred_encoded = model.predict(X_test)
    inference_time = time.time() - start_time
    
    logger.info(f"Tiempo de inferencia: {inference_time:.4f}s ({inference_time/len(X_test)*1000:.2f}ms por muestra)")
    
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    return y_pred_encoded, y_pred


def calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded):
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true (pd.Series): Target real (strings)
        y_pred (np.array): Predicciones (strings)
        y_true_encoded (np.array): Target codificado
        y_pred_encoded (np.array): Predicciones codificadas
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    logger.info("Calculando métricas...")
    
    # Métricas globales
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    f1_macro = f1_score(y_true_encoded, y_pred_encoded, average="macro")
    f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average="weighted")
    precision_macro = precision_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
    recall_macro = recall_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
    
    # Reporte por clase
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Log de métricas principales
    logger.info("="*60)
    logger.info("MÉTRICAS DE EVALUACIÓN")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    logger.info(f"Precision Macro: {precision_macro:.4f}")
    logger.info(f"Recall Macro: {recall_macro:.4f}")
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics


def log_detailed_results(report, cm):
    """
    Registra resultados detallados en logs.
    
    Args:
        report (dict): Reporte de clasificación
        cm (np.array): Matriz de confusión
    """
    logger.info("\nREPORTE POR CLASE:")
    logger.info("-" * 60)
    
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"\nClase: {class_name}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1-score']:.4f}")
            logger.info(f"  Support: {metrics['support']}")
    
    logger.info("\nMATRIZ DE CONFUSIÓN:")
    logger.info("-" * 60)
    logger.info(f"\n{cm}")


def save_results(results, reports_dir, model_name):
    """
    Guarda resultados de evaluación en JSON.
    
    Args:
        results (dict): Diccionario con todas las métricas
        reports_dir (Path): Directorio de reportes
        model_name (str): Nombre del modelo
    """
    report_path = reports_dir / f"evaluate_results_{model_name}.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en {report_path}")


def main(model_name):
    """
    Función principal del pipeline de evaluación.
    
    Args:
        model_name (str): Nombre del modelo a evaluar
    """
    try:
        # Crear directorio de logs
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"INICIANDO EVALUACIÓN: {model_name.upper()}")
        logger.info("="*60)
        
        # 1. Cargar parámetros
        params = load_params()
        
        processed_dir = Path(params["paths"]["processed_dir"])
        models_dir = Path(params["paths"]["models_dir"])
        reports_dir = Path(params["paths"]["reports_dir"])
        target_col = params["data"]["target_col"]
        
        # Crear directorio de reportes
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        # 2. Cargar datos de test
        test_path = processed_dir / "test.csv"
        X_test, y_true = load_test_data(test_path, target_col)
        
        # 3. Cargar modelo y encoder
        model, label_encoder = load_model_and_encoder(models_dir, model_name)
        
        # 4. Validar consistencia
        validate_consistency(y_true, label_encoder)
        
        # 5. Codificar target real
        y_true_encoded = label_encoder.transform(y_true)
        
        # 6. Generar predicciones
        y_pred_encoded, y_pred = make_predictions(model, X_test, label_encoder)
        
        # 7. Calcular métricas
        metrics = calculate_metrics(y_true, y_pred, y_true_encoded, y_pred_encoded)
        
        # 8. Log detallado
        log_detailed_results(metrics["classification_report"], metrics["confusion_matrix"])
        
        # 9. Preparar resultados finales
        results = {
            "model": model_name,
            "test_samples": len(y_true),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"]
            },
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"]
        }
        
        # 10. Guardar resultados
        save_results(results, reports_dir, model_name)

        log_evaluation_to_run(
            model_name=model_name,
            results=results,
            reports_dir=reports_dir
        )

        
        logger.info("="*60)
        logger.info("EVALUACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error crítico en evaluate.py: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación final del modelo"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "randomforest", "xgboost"],
        required=True,
        help="Modelo a evaluar: logistic, randomforest, xgboost"
    )
    args = parser.parse_args()
    
    main(args.model)