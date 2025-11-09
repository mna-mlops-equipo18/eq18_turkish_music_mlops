FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN fallocate -l 4G /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile && \
    pip install --no-cache-dir torch pandas numpy scikit-learn && \
    pip install --no-cache-dir -r requirements.txt && \
    swapoff /swapfile && \
    rm /swapfile

COPY ./models /app/models

COPY ./eq18_turkish_music_mlops /app/eq18_turkish_music_mlops

COPY ./api.py /app/api.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]