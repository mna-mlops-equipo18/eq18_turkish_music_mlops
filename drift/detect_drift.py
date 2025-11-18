"""
drift/detect_drift.py

Script para detectar *data drift* entre:

    - Dataset de referencia: data/processed/test.csv
    - Dataset actual (con drift): data/drift/test_drift_<scenario>.csv

Escenarios usados en el proyecto:
    - 'energy' → drift en _RMSenergy_Mean (angry, happy)
    - 'tempo'  → drift en _Tempo_Mean (happy, angry, sad)

Herramientas de detección:
    1) Evidently (DataDriftPreset)
    2) Tests estadísticos clásicos:
        - Kolmogórov–Smirnov (KS) para columnas numéricas
        - Chi-cuadrado para la distribución de la variable Class
    3) Clasificador de dominio (referencia vs actual) con Logistic Regression

Ejemplo de uso:
    python drift/detect_drift.py --scenario energy
    python drift/detect_drift.py --scenario tempo
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------
# IMPORTS DE EVIDENTLY (versión 0.7.x)
# ---------------------------------------------------------
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------
# Configuración de logging global
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Rutas del proyecto
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]                  # Carpeta raíz del proyecto
REF_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"         # Dataset de referencia
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"                         # Carpeta de CSV con drift
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"                    # Carpeta para reportes de drift
REPORTS_DIR.mkdir(parents=True, exist_ok=True)                      # Crear carpeta si no existe


# -------------------------------------------------------------------------
# Utilidad: identificar columnas numéricas
# -------------------------------------------------------------------------
def get_numerical_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """
    Devuelve la lista de columnas numéricas del DataFrame, excluyendo
    las de exclude_cols (por ejemplo, 'Class').
    """
    if exclude_cols is None:
        exclude_cols = []

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    return num_cols


# -------------------------------------------------------------------------
# Herramienta 1: Evidently (DataDriftPreset)
# -------------------------------------------------------------------------
def run_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    scenario: str,
) -> dict:
    """
    Ejecuta Evidently con el DataDriftPreset entre referencia y actual.

    Genera un JSON completo de Evidently en:
        reports/drift/evidently_drift_<scenario>.json
    """
    report = Report(metrics=[DataDriftPreset()])

    # En Evidently 0.7.x, run devuelve un "snapshot" con métodos .save_json()
    snapshot = report.run(current, reference)

    evidently_json_path = REPORTS_DIR / f"evidently_drift_{scenario}.json"
    snapshot.save_json(str(evidently_json_path))

    summary = {
        "evidently_json_file": str(evidently_json_path.relative_to(PROJECT_ROOT)),
        "note": "El detalle de columnas con drift está en el JSON de Evidently.",
    }
    return summary


# -------------------------------------------------------------------------
# Herramienta 2: Tests estadísticos (KS y Chi-cuadrado)
# -------------------------------------------------------------------------
def run_statistical_tests(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """
    Ejecuta:
        - Kolmogórov–Smirnov (KS) para columnas numéricas
        - Chi-cuadrado para la distribución de la variable 'Class'
    """
    results = {}

    # Alinear columnas para comparar manzana con manzana
    common_cols = reference.columns.intersection(current.columns)
    reference = reference[common_cols].copy()
    current = current[common_cols].copy()

    # --- KS para columnas numéricas ---
    num_cols = get_numerical_columns(reference, exclude_cols=["Class"])
    ks_results = []
    drifted_cols = []

    for col in num_cols:
        ref_col = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur_col = pd.to_numeric(current[col], errors="coerce").dropna()

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        stat, pvalue = ks_2samp(ref_col, cur_col)

        ks_results.append(
            {
                "column": col,
                "ks_stat": float(stat),
                "p_value": float(pvalue),
                "drift_detected": bool(pvalue < alpha),
            }
        )

        if pvalue < alpha:
            drifted_cols.append(col)

    results["ks_results"] = ks_results
    results["ks_drifted_columns"] = drifted_cols

    # --- Chi-cuadrado para la distribución de 'Class' ---
    if "Class" in reference.columns:
        ref_counts = reference["Class"].value_counts().sort_index()
        cur_counts = current["Class"].value_counts().sort_index()

        all_classes = sorted(set(ref_counts.index) | set(cur_counts.index))
        ref_arr = np.array([ref_counts.get(c, 0) for c in all_classes])
        cur_arr = np.array([cur_counts.get(c, 0) for c in all_classes])

        contingency = np.vstack([ref_arr, cur_arr])

        chi2_stat, chi2_p, dof, _ = chi2_contingency(contingency)

        results["class_chi2"] = {
            "classes": all_classes,
            "chi2_stat": float(chi2_stat),
            "p_value": float(chi2_p),
            "dof": int(dof),
            "drift_detected": bool(chi2_p < alpha),
        }
    else:
        results["class_chi2"] = {
            "error": "No se encontró la columna 'Class' para el test Chi-cuadrado."
        }

    return results


# -------------------------------------------------------------------------
# Herramienta 3: Clasificador de dominio (ref vs actual)
# -------------------------------------------------------------------------
def run_domain_classifier(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Entrena un clasificador binario para distinguir:

        - dominio 0: referencia
        - dominio 1: actual (con drift)

    Si accuracy/AUC es muy alta (p.ej. > 0.8), hay drift fuerte.
    """
    common_cols = reference.columns.intersection(current.columns)
    reference = reference[common_cols].copy()
    current = current[common_cols].copy()

    feature_cols = get_numerical_columns(reference, exclude_cols=["Class"])
    if not feature_cols:
        return {"error": "No se encontraron columnas numéricas para el clasificador de dominio."}

    X_ref = reference[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_cur = current[feature_cols].apply(pd.to_numeric, errors="coerce")

    y_ref = np.zeros(len(X_ref), dtype=int)
    y_cur = np.ones(len(X_cur), dtype=int)

    X = pd.concat([X_ref, X_cur], axis=0)
    y = np.concatenate([y_ref, y_cur], axis=0)

    # Quitar filas con NaN
    mask_valid = ~X.isna().any(axis=1)
    X = X[mask_valid]
    y = y[mask_valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = None

    result = {
        "accuracy": float(acc),
        "auc": float(auc) if auc is not None else None,
        "drift_detected_strong": bool(acc > 0.8 or (auc is not None and auc > 0.8)),
    }
    return result


# -------------------------------------------------------------------------
# Función principal: orquesta las 3 herramientas y guarda resumen
# -------------------------------------------------------------------------
def main(scenario: str = "energy") -> None:
    """
    Carga el dataset de referencia y el dataset con drift para el escenario
    indicado, corre las 3 herramientas de detección y guarda un resumen:

        reports/drift/detect_summary_<scenario>.json
    """
    reference_path = REF_PATH
    current_path = DRIFT_DIR / f"test_drift_{scenario}.csv"

    logger.info("Cargando dataset de referencia: %s", reference_path)
    logger.info("Cargando dataset actual con drift (%s): %s", scenario, current_path)

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    # Alinear columnas
    common_cols = reference_df.columns.intersection(current_df.columns)
    reference_df = reference_df[common_cols].copy()
    current_df = current_df[common_cols].copy()

    # 1) Evidently
    logger.info("Ejecutando Evidently (DataDriftPreset)...")
    evidently_summary = run_evidently_drift(
        reference=reference_df,
        current=current_df,
        scenario=scenario,
    )

    # 2) Tests estadísticos
    logger.info("Ejecutando tests estadísticos (KS + Chi-cuadrado)...")
    stats_summary = run_statistical_tests(
        reference=reference_df,
        current=current_df,
        alpha=0.05,
    )

    # 3) Clasificador de dominio
    logger.info("Entrenando clasificador de dominio (referencia vs actual)...")
    domain_summary = run_domain_classifier(
        reference=reference_df,
        current=current_df,
    )

    # 4) Resumen unificado
    summary = {
        "scenario": scenario,
        "evidently": evidently_summary,
        "statistical_tests": stats_summary,
        "domain_classifier": domain_summary,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / f"detect_summary_{scenario}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Resumen de detección de drift guardado en: %s", summary_path)

    # 5) RESUMEN AMIGABLE PARA PANTALLAZO
    logger.info("============================================================")
    logger.info("RESUMEN DE DETECCIÓN DE DRIFT - ESCENARIO: %s", scenario)
    logger.info("============================================================")

    ks_results = summary["statistical_tests"]["ks_results"]
    ks_drifted = summary["statistical_tests"]["ks_drifted_columns"]

    logger.info("Columnas con drift según KS (p < 0.05): %s", ks_drifted)

    # Columna "principal" a destacar según escenario
    highlight_col = None
    if scenario == "energy":
        highlight_col = "_RMSenergy_Mean"
    elif scenario == "tempo":
        # Buscar la columna de tempo más evidente en los resultados KS
        tempo_entry = next(
            (
                item
                for item in ks_results
                if "tempo" in item["column"].lower() or "bpm" in item["column"].lower()
            ),
            None,
        )
        if tempo_entry is not None:
            highlight_col = tempo_entry["column"]

    if highlight_col is not None:
        entry = next(
            (item for item in ks_results if item["column"] == highlight_col),
            None,
        )
        if entry is not None:
            logger.info(
                "KS para %s -> stat=%.4f, p_value=%.4f, drift=%s",
                highlight_col,
                entry["ks_stat"],
                entry["p_value"],
                entry["drift_detected"],
            )
        else:
            logger.info(
                "No se encontraron resultados KS para la columna esperada: %s",
                highlight_col,
            )
    else:
        logger.info(
            "No se definió una columna destacada para el escenario '%s'.",
            scenario,
        )

    # Chi-cuadrado sobre distribución de clases
    chi2 = summary["statistical_tests"]["class_chi2"]
    if "error" not in chi2:
        logger.info(
            "Chi² clases -> chi2_stat=%.4f, p_value=%.4f, drift_detected=%s",
            chi2["chi2_stat"],
            chi2["p_value"],
            chi2["drift_detected"],
        )
    else:
        logger.info("Chi² clases -> %s", chi2["error"])

    # Clasificador de dominio
    dom = summary["domain_classifier"]
    if "error" not in dom:
        logger.info(
            "Clasificador de dominio -> accuracy=%.4f, AUC=%.4f, drift_fuerte=%s",
            dom.get("accuracy", float("nan")),
            dom.get("auc", float("nan")),
            dom.get("drift_detected_strong", False),
        )
    else:
        logger.info("Clasificador de dominio -> %s", dom["error"])

    logger.info("============================================================")
    logger.info("Detección de drift completada para escenario '%s'.", scenario)
    logger.info("============================================================")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección de data drift entre test.csv y test_drift_<scenario>.csv",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="energy",
        help="Nombre del escenario de drift: 'energy' o 'tempo'.",
    )
    args = parser.parse_args()

    main(scenario=args.scenario)
