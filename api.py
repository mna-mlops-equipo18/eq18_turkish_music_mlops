import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from pathlib import Path

# (CRÍTICO) Importa tus transformers personalizados
from eq18_turkish_music_mlops.utils.transformers import clean_finite_values, OutlierIQRTransformer

# --- Cargar Artefactos ---
MODEL_PATH = Path("models/model_randomforest.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
model = None
label_encoder = None

def load_artifacts():
    """Carga los artefactos del modelo en memoria."""
    global model, label_encoder
    try:
        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            print("Modelo y Label Encoder cargados exitosamente.")
        else:
            print(f"Error: No se encontraron los archivos en {MODEL_PATH} o {ENCODER_PATH}")
    except Exception as e:
        print(f"Error al cargar artefactos: {e}")

# --- Esquemas de Pydantic ---
class FeaturesPayload(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    emotion_predicted: str
    model_version: str = "randomforest_v1"

# --- Inicializar la App de FastAPI ---
app = FastAPI(
    title="API de Clasificación de Emociones",
    description="Servicio de MLOps para predicción de emociones musicales.",
    version="1.0"
)

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/health", status_code=200)
def health_check():
    """Endpoint para que Docker verifique que la API está viva."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(payload: FeaturesPayload):
    """
    Endpoint de predicción. 
    Recibe un JSON con las 50 features y devuelve la emoción.
    """
    if model is None:
        return {"error": "Modelo no cargado. Revisa los logs del servidor."}

    input_df = pd.DataFrame([payload.features])
    pred_encoded = model.predict(input_df)
    pred_class = label_encoder.inverse_transform(pred_encoded)
    return PredictionResponse(emotion_predicted=pred_class[0])

@app.get("/")
def read_root():
    return {"status": "API de modelo en línea. Visita /docs para probar."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)