FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install system dependencies needed for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "fastdemo:app", "--host", "0.0.0.0", "--port", "8080"]