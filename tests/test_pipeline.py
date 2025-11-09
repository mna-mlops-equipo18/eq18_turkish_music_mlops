import pytest
import joblib
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eq18_turkish_music_mlops.utils.transformers import OutlierIQRTransformer, clean_finite_values

MODEL_PATH = Path("models/model_randomforest.pkl") 
ENCODER_PATH = Path("models/label_encoder.pkl")
EXPECTED_CLASSES = {'angry', 'happy', 'relax', 'sad'} 

@pytest.fixture(scope="module")
def sample_data():
    """
    Crea un dato de prueba (un diccionario con tus 50 features).
    IMPORTANTE: Debes reemplazar esto con 50 features reales de tus datos.
    Por ahora, rellenamos con valores de ejemplo y ceros.
    """
    features = {
        "tempo": 120.0,
        "beats": 50.0,
        "chroma_stft": 0.4,
        "rmse": 0.05,
        "spectral_centroid": 2000.0,
        "spectral_bandwidth": 1500.0,
        "rolloff": 4000.0,
        "zero_crossing_rate": 0.08,
        "mfcc1": -150.0,
        "mfcc2": 100.0,
        **{f"feature_{i}": 0.0 for i in range(11, 51)}
    }
    # Asegurarnos de que tenemos exactamente 50 features
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
    Valida que (datos) -> (pipeline) -> (predicción) funciona.
    """
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        pytest.skip("Saltando prueba de predicción porque los modelos no fueron encontrados.")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    input_df = pd.DataFrame([sample_data])
    
    pred_encoded = model.predict(input_df)
    
    pred_class = encoder.inverse_transform(pred_encoded)
    
    assert pred_class[0] in EXPECTED_CLASSES, f"La predicción '{pred_class[0]}' no es una clase válida."
    print(f"\nPrueba de integración exitosa: predicción = {pred_class[0]}")

