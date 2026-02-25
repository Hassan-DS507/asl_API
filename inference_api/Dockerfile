FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# أضفنا --no-install-recommends ومسحنا الكاش عشان نصغر الحجم
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD sh -c "uvicorn fastdemo:app --host 0.0.0.0 --port ${PORT:-8080}"
