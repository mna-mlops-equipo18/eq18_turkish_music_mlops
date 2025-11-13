FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore
COPY dvc.yaml /app/dvc.yaml
COPY dvc.lock /app/dvc.lock

RUN test -f dvc.yaml || { echo "ERROR: dvc.yaml no encontrado"; exit 1; }
RUN test -f dvc.lock || { echo "ERROR: dvc.lock no encontrado"; exit 1; }

ARG AZURE_CONTAINER_NAME
ARG AZURE_STORAGE_ACCOUNT

RUN --mount=type=secret,id=AZURE_PROJECT_KEY \
    if [ ! -f /run/secrets/AZURE_PROJECT_KEY ]; then \
        echo "ERROR: Secret AZURE_PROJECT_KEY no encontrado"; \
        exit 1; \
    fi && \
    export AZURE_STORAGE_KEY=$(cat /run/secrets/AZURE_PROJECT_KEY) && \
    dvc remote modify azure-storage account_key "$AZURE_STORAGE_KEY" --local && \
    echo "Iniciando descarga de artefactos DVC..." && \
    dvc pull -v prepare train_logistic train_randomforest train_xgboost && \
    echo "✓ Artefactos descargados exitosamente"

RUN test -d models || { echo "ERROR: Directorio models/ no existe después de dvc pull"; exit 1; }
RUN test -d data/processed || { echo "ERROR: Directorio data/processed/ no existe después de dvc pull"; exit 1; }

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /app/models /app/models
COPY --from=builder /app/data/processed /app/data/processed

COPY ./api.py /app/api.py
COPY ./eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]