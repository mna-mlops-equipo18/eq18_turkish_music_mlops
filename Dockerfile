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

RUN python -c "import dvc_azure; import torch; print('✓ Dependencias OK')"

COPY .dvc /app/.dvc
COPY dvc.yaml /app/dvc.yaml
COPY dvc.lock /app/dvc.lock

RUN --mount=type=secret,id=AZURE_PROJECT_KEY \
    export AZURE_KEY=$(cat /run/secrets/AZURE_PROJECT_KEY) && \
    dvc remote modify azure-storage account_key "$AZURE_KEY" --local && \
    dvc pull -v prepare train_logistic train_randomforest train_xgboost

RUN test -d models && test -d data/processed

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/models /app/models
COPY --from=builder /app/data/processed /app/data/processed

COPY ./api.py /app/api.py
COPY ./eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]