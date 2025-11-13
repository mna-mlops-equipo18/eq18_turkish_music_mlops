FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch pandas numpy scikit-learn
RUN pip install --no-cache-dir "dvc[azure]" mlflow xgboost

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN --mount=type=secret,id=AZURE_PROJECT_KEY \
    dvc remote modify azure-storage account_key "$(cat /run/secrets/AZURE_PROJECT_KEY)" --local && \
    dvc pull -f \
        prepare \
        train_logistic \
        train_randomforest \
        train_xgboost

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models /app/models
COPY --from=builder /app/data/processed /app/data/processed
COPY --from=builder /app/api.py /app/api.py
COPY --from=builder /app/eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
