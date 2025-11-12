FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    fallocate -l 4G /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

COPY requirements.txt .
RUN pip install --no-cache-dir torch pandas numpy scikit-learn
RUN pip install --no-cache-dir "dvc[azure]" mlflow xgboost
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG AZURE_PROJECT_KEY
RUN dvc remote modify azure-storage account_key "$AZURE_PROJECT_KEY" --local && \
    dvc pull -f \
      prepare \
      train_logistic \
      train_randomforest \
      train_xgboost && \
    ...

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/models /app/models

COPY ./api.py /app/api.py
COPY ./eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]