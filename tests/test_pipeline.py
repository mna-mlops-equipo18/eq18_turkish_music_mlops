# Archivo: tests/test_pipeline.py

import pytest
import joblib
import pandas as pd
import sys
import os
from pathlib import Path

# --- Configuración Clave ---
# 1. Añade el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. (CRÍTICO) Importa tus transformers personalizados
# 'joblib.load()' los necesita en el scope para poder cargar el .pkl
from eq18_turkish_music_mlops.utils.transformers import OutlierIQRTransformer, clean_finite_values

# --- Configuración de la Prueba ---
MODEL_PATH = Path("models/model_randomforest.pkl") 
ENCODER_PATH = Path("models/label_encoder.pkl")
EXPECTED_CLASSES = {'angry', 'happy', 'relax', 'sad'}

@pytest.fixture(scope="module")
def sample_data():
    """
    Crea un dato de prueba (un diccionario) usando los 50 NOMBRES DE COLUMNA REALES.
    """
    features = {
        '_MFCC_Mean_10': 0.0, '_AttackTime_Mean': 0.0, '_Roughness_Slope': 0.0,
        '_Chromagram_Mean_6': 0.0, '_Chromagram_Mean_5': 0.0, '_HarmonicChangeDetectionFunction_Std': 0.0,
        '_RMSenergy_Mean': 0.0, '_HarmonicChangeDetectionFunction_Slope': 0.0, '_HarmonicChangeDetectionFunction_Mean': 0.0,
        '_Pulseclarity_Mean': 0.0, '_Chromagram_Mean_11': 0.0, '_Chromagram_Mean_2': 0.0,
        '_MFCC_Mean_5': 0.0, '_Lowenergy_Mean': 0.0, '_Rolloff_Mean': 0.0,
        '_MFCC_Mean_12': 0.0, '_Spectralspread_Mean': 0.0, '_Roughness_Mean': 0.0,
        '_Chromagram_Mean_7': 0.0, '_Chromagram_Mean_8': 0.0, '_Chromagram_Mean_3': 0.0,
        '_Spectralskewness_Mean': 0.0, '_Tempo_Mean': 0.0, '_Chromagram_Mean_4': 0.0,
        '_MFCC_Mean_7': 0.0, '_MFCC_Mean_9': 0.0, '_MFCC_Mean_1': 0.0,
        '_MFCC_Mean_4': 0.0, '_Zero-crossingrate_Mean': 0.0, '_Spectralkurtosis_Mean': 0.0,
        '_Chromagram_Mean_10': 0.0, '_MFCC_Mean_11': 0.0, '_MFCC_Mean_3': 0.0,
        '_EntropyofSpectrum_Mean': 0.0, '_Brightness_Mean': 0.0, '_MFCC_Mean_8': 0.0,
        '_MFCC_Mean_13': 0.0, '_HarmonicChangeDetectionFunction_PeriodEntropy': 0.0, '_Chromagram_Mean_12': 0.0,
        '_AttackTime_Slope': 0.0, '_Chromagram_Mean_1': 0.0, 
        '_HarmonicChangeDetectionFunction_PeriodFreq': 0.0, '_HarmonicChangeDetectionFunction_PeriodAmp': 0.0,
        '_Fluctuation_Mean': 0.0, '_MFCC_Mean_2': 0.0, '_Eventdensity_Mean': 0.0,
        '_Spectralflatness_Mean': 0.0, '_Chromagram_Mean_9': 0.0, '_MFCC_Mean_6': 0.0,
        '_Spectralcentroid_Mean': 0.0
    }
    
    assert len(features) == 50, "El dato de prueba no tiene 50 features"
    return features


def test_model_loading():
    """Prueba que los artefactos del modelo (pkl) existan y se puedan cargar."""
    assert MODEL_PATH.exists(), f"El archivo del modelo no se encontró en {MODEL_PATH}. ¿Corriste 'dvc pull models'?"
    assert ENCODER_PATH.exists(), f"El archivo del encoder no se encontró en {ENCODER_PATH}. ¿Corriste 'dvc pull models'?"
    
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        assert model is not None, "El modelo cargado es None"
        assert encoder is not None, "El encoder cargado es None"
    except Exception as e:
        pytest.fail(f"Fallo al cargar los modelos .pkl: {e}. (¿Importaste los transformers personalizados?)")

def test_prediction_integration(sample_data):
    """
    Prueba de integración: carga el pipeline y realiza una predicción.
    """
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        pytest.skip("Saltando prueba de predicción porque los modelos no fueron encontrados.")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    input_df = pd.DataFrame([sample_data])
    pred_encoded = model.predict(input_df)
    pred_class = encoder.inverse_transform(pred_encoded)
    
    assert pred_class[0] in EXPECTED_CLASSES, f"La predicción '{pred_class[0]}' no es una clase válida."