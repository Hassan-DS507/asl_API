FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn fastdemo:app --host 0.0.0.0 --port $PORT
