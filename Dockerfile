# Use high-performance Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    DEEPFACE_HOME=/app/models \
    HF_HOME=/app/models/huggingface \
    HOME=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create model directory and set permissions
RUN mkdir -p /app/models && chmod -R 777 /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]