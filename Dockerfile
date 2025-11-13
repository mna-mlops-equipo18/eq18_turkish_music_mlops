FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "dvc[azure]"

COPY requirements.txt .
RUN grep -v "^torch" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements_filtered.txt

RUN python -c "import dvc_azure; import torch; print('Dependencias OK')"

COPY . .

RUN --mount=type=secret,id=AZURE_PROJECT_KEY \
    set -e && \
    echo "🔧 Configurando DVC con Azure..." && \
    dvc remote modify azure-storage account_key "$(cat /run/secrets/AZURE_PROJECT_KEY)" --local && \
    echo "  Descargando artefactos DVC..." && \
    dvc pull -v prepare train_logistic train_randomforest train_xgboost && \
    echo " Artefactos descargados exitosamente"

RUN test -d models || { echo " ERROR: models/ no existe"; exit 1; } && \
    test -d data/processed || { echo " ERROR: data/processed/ no existe"; exit 1; } && \
    echo " Artefactos validados correctamente"

FROM python:3.11-slim

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
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]