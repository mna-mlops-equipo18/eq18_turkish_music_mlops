import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from pathlib import Path

# cargamos el modelo
MODEL_PATH = Path("models/model_randomforest.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
model = None
label_encoder = None

if MODEL_PATH.exists() and ENCODER_PATH.exists():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Modelo y Label Encoder cargados exitosamente.")
else:
    print(f"Error: No se encontraron los archivos en {MODEL_PATH} o {ENCODER_PATH}")
    print("Asegúrate de ejecutar 'dvc pull' antes de iniciar la API.")

# cargamos las features a un diccionario
class FeaturesPayload(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    emotion_predicted: str
    model_version: str = "randomforest_v1"

app = FastAPI(
    title="API de Clasificación de Emociones",
    description="Servicio de prueba para exponer el modelo de ML (Rúbrica 4)",
    version="1.0"
)

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(payload: FeaturesPayload):
    """
    Endpoint de predicción. 
    Recibimos 50 features y mandamos json
    """
    if model is None:
        return {"error": "Modelo no cargado."}

    input_df = pd.DataFrame([payload.features])

    pred_encoded = model.predict(input_df)
    
    pred_class = label_encoder.inverse_transform(pred_encoded)

    return PredictionResponse(emotion_predicted=pred_class[0])

@app.get("/")
def read_root():
    return {"status": "API Funcionando"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)