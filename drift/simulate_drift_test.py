import argparse
import logging
from pathlib import Path

import numpy as np  # Se mantiene por si luego agregan ruido o más escenarios
import pandas as pd

# -------------------------------------------------------------------------
# Rutas base del proyecto
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]               # Carpeta raíz
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"     # Test base (sin drift)
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"                      # Carpeta para CSV con drift
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Utilidad: buscar columnas por palabra clave en el nombre
# -------------------------------------------------------------------------
def find_columns_by_keywords(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    """
    Devuelve columnas cuyo nombre contiene alguna de las palabras clave.
    Se usa para localizar columnas como '_Tempo_Mean', 'bpm', etc.
    """
    cols = [
        c
        for c in df.columns
        if any(k.lower() in c.lower() for k in keywords)
    ]
    return cols


# -------------------------------------------------------------------------
# Escenario 1: Drift de energía (RMS) en clases de alta activación
# -------------------------------------------------------------------------
def apply_energy_drift(df: pd.DataFrame, factor: float = 1.6) -> pd.DataFrame:
    """
    Aumenta la energía RMS media (_RMSenergy_Mean) en clases 'angry' y 'happy'.

    factor > 1 → hace el cambio más fuerte (ej. 1.6 = +60%).
    """
    df_drift = df.copy()

    energy_col = "_RMSenergy_Mean"
    if energy_col not in df_drift.columns:
        raise ValueError(f"La columna '{energy_col}' no existe en el dataset de test.")

    # Asegurar tipo numérico para evitar errores silenciosos
    df_drift[energy_col] = pd.to_numeric(df_drift[energy_col], errors="coerce")

    # Más energía solo para clases de alta activación
    mask_high = df_drift["Class"].isin(["angry", "happy"])

    logging.info(
        "Escenario 'energy': multiplicando %s por %.2f en clases ['angry','happy']",
        energy_col,
        factor,
    )
    df_drift.loc[mask_high, energy_col] = (
        df_drift.loc[mask_high, energy_col] * factor
    )

    return df_drift


# -------------------------------------------------------------------------
# Escenario 2: Drift en tempo/ritmo
# -------------------------------------------------------------------------
def apply_tempo_drift(df: pd.DataFrame, factor: float = 1.25) -> pd.DataFrame:
    """
    Modifica las características de tempo:

        - Aumenta tempo en 'happy' y 'angry' (× factor).
        - Disminuye tempo en 'sad' (÷ factor).
        - 'relax' se mantiene sin cambio explícito.

    Las columnas de tempo se localizan por nombre: contiene 'tempo' o 'bpm'.
    """
    df_drift = df.copy()

    tempo_cols = find_columns_by_keywords(df_drift, keywords=["tempo", "bpm"])
    if not tempo_cols:
        logging.warning(
            "Escenario 'tempo': no se encontraron columnas de tempo por nombre."
        )
        return df_drift

    logging.info(
        "Escenario 'tempo': usando factor %.2f en columnas %s",
        factor,
        tempo_cols,
    )

    # Asegurar tipo numérico
    df_drift[tempo_cols] = df_drift[tempo_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    mask_happy_angry = df_drift["Class"].isin(["happy", "angry"])
    mask_sad = df_drift["Class"] == "sad"

    # Acelerar happy/angry
    df_drift.loc[mask_happy_angry, tempo_cols] = (
        df_drift.loc[mask_happy_angry, tempo_cols] * factor
    )
    # Ralentizar sad
    df_drift.loc[mask_sad, tempo_cols] = (
        df_drift.loc[mask_sad, tempo_cols] / factor
    )

    return df_drift


# -------------------------------------------------------------------------
# Selector de escenario
# -------------------------------------------------------------------------
def apply_drift(df: pd.DataFrame, scenario: str, factor: float) -> pd.DataFrame:
    """
    Selecciona y aplica el tipo de drift solicitado.

    - scenario = 'energy' → escala energía RMS en angry/happy.
    - scenario = 'tempo'  → modifica tempo para happy/angry/sad.
    """
    scenario = scenario.lower()

    if scenario == "energy":
        return apply_energy_drift(df, factor=factor)
    if scenario == "tempo":
        return apply_tempo_drift(df, factor=factor)

    raise NotImplementedError(
        f"Escenario '{scenario}' no reconocido. Usa: energy o tempo."
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(scenario: str = "energy", factor: float = 1.6) -> None:
    """
    Lee data/processed/test.csv, aplica el drift indicado y guarda
    un nuevo CSV en data/drift/ con el nombre:

        test_drift_<scenario>.csv
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )

    logging.info("Leyendo dataset base de test: %s", TEST_PATH)
    df_test = pd.read_csv(TEST_PATH)

    logging.info("Aplicando escenario de drift: %s (factor=%.2f)", scenario, factor)
    df_drift = apply_drift(df_test, scenario=scenario, factor=factor)

    output_path = DRIFT_DIR / f"test_drift_{scenario.lower()}.csv"
    logging.info("Guardando dataset con drift en: %s", output_path)

    df_drift.to_csv(output_path, index=False)
    logging.info("Drift simulado correctamente. Filas: %d, Columnas: %d", *df_drift.shape)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simular data drift en el conjunto de test (escenarios energy/tempo)."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="energy",
        help="Escenario de drift: 'energy' o 'tempo'.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.6,
        help=(
            "Intensidad del drift. "
            "energy: escala de energía; "
            "tempo: escala de tempo."
        ),
    )

    args = parser.parse_args()
    main(scenario=args.scenario, factor=args.factor)
