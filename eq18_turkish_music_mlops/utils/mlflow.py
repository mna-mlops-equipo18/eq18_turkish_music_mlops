"""

Módulo de MLFLOW.
Dentro de este archivo, tendremos todas las funciones que necesitaremos
para hacer uso de MLFLOW en cualquier parte del proyecto

"""

import mlflow
import mlflow.sklearn
import logging
import json
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def start_training_run(
    model_name: str,
    params: Dict[str, Any],
    grid: GridSearchCV,
    models_dir: Path,
    reports_dir: Path
):
    """
    Inicia un nuevo run en MLflow para una etapa de entrenamiento.

    Registra:
    - Parámetros de procesamiento y mejores hiperparámetros.
    - Métricas de validación cruzada (CV).
    - Artefactos (modelo .pkl, encoder .pkl, reporte .json).
    - Modelo en formato nativo de MLflow.
    - Guarda el run_id en un archivo para la etapa de evaluación.
    
    Args:
        model_name (str): Nombre del modelo (e.g., 'logistic').
        params (dict): Diccionario de 'params.yaml'.
        grid (GridSearchCV): Objeto GridSearchCV ya ajustado.
        models_dir (Path): Directorio de modelos.
        reports_dir (Path): Directorio de reportes.
    """
    logger.info(f"Inicia MLFLOW [train_{model_name}]")
    
    try:
        # 1. Configurar experimento
        try:
            mlflow.set_experiment(params["mlflow"]["experiment_name"])
        except KeyError:
            logger.warning("No se definió 'mlflow.experiment_name' en params.yaml. Usando 'default'.")
            mlflow.set_experiment("turkish_music_emotion_default")

        # 2. Iniciar el run
        with mlflow.start_run(run_name=f"train_{model_name}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            # 3. Loguear Parámetros
            mlflow.log_param("model_name", model_name)
            if "processing" in params:
                mlflow.log_params(params["processing"])
                
            # Limpiar prefijo 'modelo__' de los hiperparámetros
            best_params_cleaned = {k.replace("modelo__", ""): v for k, v in grid.best_params_.items()}
            mlflow.log_params(best_params_cleaned)
            logger.info("Parámetros logueados en MLflow.")

            # 4. Loguear Métricas (de CV)
            mlflow.log_metric("cv_score_mean", grid.best_score_)
            mlflow.log_metric("cv_score_std", grid.cv_results_['std_test_score'][grid.best_index_])
            logger.info("Métricas de CV logueadas en MLflow.")

            # 5. Loguear Artefactos (archivos generados por DVC)
            model_path = models_dir / f"model_{model_name}.pkl"
            encoder_path = models_dir / "label_encoder.pkl"
            report_path = reports_dir / f"train_results_{model_name}.json"
            
            mlflow.log_artifact(str(model_path), "model_files")
            if encoder_path.exists():
                mlflow.log_artifact(str(encoder_path), "model_files")
            mlflow.log_artifact(str(report_path), "reports")
            logger.info("Artefactos de DVC logueados en MLflow.")

            # 6. Loguear Modelo (formato nativo MLflow)
            mlflow.sklearn.log_model(
                sk_model=grid.best_estimator_,
                artifact_path=f"mlflow_model_{model_name}"
            )
            logger.info("Modelo logueado en formato nativo de MLflow.")

            # 7. Guardar Run ID para la siguiente etapa
            run_id_path = reports_dir / f"run_id_{model_name}.txt"
            with open(run_id_path, "w", encoding="utf-8") as f:
                f.write(run_id)
            logger.info(f"Run ID ({run_id}) guardado en {run_id_path}")

    except Exception as e:
        logger.error(f"Error al loguear entrenamiento en MLflow: {e}", exc_info=True)
        # No relanzamos la excepción para no fallar el pipeline de DVC
        # si MLflow (que es opcional) falla.


def log_evaluation_to_run(
    model_name: str,
    results: Dict[str, Any],
    reports_dir: Path
):
    """
    Reanuda un run existente de MLflow para loguear métricas de evaluación.

    Carga el run_id desde el archivo, reabre el run y añade:
    - Métricas finales (accuracy, f1_macro, etc.).
    - Artefacto de reporte de evaluación (.json).
    
    Args:
        model_name (str): Nombre del modelo (e.g., 'logistic').
        results (dict): Diccionario de resultados de evaluate.py.
        reports_dir (Path): Directorio de reportes.
    """
    logger.info(f"Logueando métricas de evaluación en MLflow para [{model_name}]")
    
    run_id_path = reports_dir / f"run_id_{model_name}.txt"
    
    try:
        # 1. Cargar el Run ID
        if not run_id_path.exists():
            logger.warning(f"No se encontró el archivo Run ID ({run_id_path}). "
                           f"No se loguearán métricas a MLflow.")
            return

        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
            
        if not run_id:
            logger.warning(f"El archivo Run ID ({run_id_path}) está vacío.")
            return

        logger.info(f"Reabriendo MLflow Run ID: {run_id}")

        # 2. Reabrir el run
        with mlflow.start_run(run_id=run_id):
            
            # 3. Loguear Métricas (finales, del test set)
            if "metrics" in results:
                mlflow.log_metrics(results["metrics"])
                logger.info("Métricas finales (test) logueadas en MLflow.")
            
            # 4. Loguear Artefacto (reporte de evaluación)
            report_path = reports_dir / f"evaluate_results_{model_name}.json"
            if report_path.exists():
                mlflow.log_artifact(str(report_path), "reports")
                logger.info("Reporte de evaluación logueado en MLflow.")
                
            # (Aquí iría el código para loguear la Matriz de Confusión)

    except Exception as e:
        logger.error(f"Error al loguear evaluación en MLflow: {e}", exc_info=True)