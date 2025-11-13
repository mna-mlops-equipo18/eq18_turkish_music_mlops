# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar git para DVC
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python CON dvc[azure]
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "dvc[azure]" && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import dvc_azure; print('✓ dvc-azure instalado')"

# Copiar archivos DVC
COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore
COPY dvc.yaml /app/dvc.yaml
COPY dvc.lock /app/dvc.lock

# Configurar DVC y descargar artefactos
RUN --mount=type=secret,id=AZURE_PROJECT_KEY \
    if [ ! -f /run/secrets/AZURE_PROJECT_KEY ]; then \
        echo "❌ ERROR: Secret no encontrado"; \
        exit 1; \
    fi && \
    export AZURE_KEY=$(cat /run/secrets/AZURE_PROJECT_KEY) && \
    dvc remote modify azure-storage account_key "$AZURE_KEY" --local && \
    echo "Descargando artefactos..." && \
    dvc pull -v prepare train_logistic train_randomforest train_xgboost && \
    echo "✓ Artefactos descargados"

# Validar modelos
RUN test -d models || { echo "❌ ERROR: models/ no existe"; exit 1; }
RUN test -d data/processed || { echo "❌ ERROR: data/processed/ no existe"; exit 1; }

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copiar artefactos
COPY --from=builder /app/models /app/models
COPY --from=builder /app/data/processed /app/data/processed

# Copiar código
COPY ./api.py /app/api.py
COPY ./eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

# Usuario no privilegiado
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]