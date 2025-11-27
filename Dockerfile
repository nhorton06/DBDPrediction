# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    FLASK_HOST=0.0.0.0 \
    FLASK_DEBUG=False

# Install system dependencies and upgrade all packages (including tar) to latest versions
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files from src folder
COPY src/app.py .
COPY src/train_model.py .
COPY src/save_model.py .
COPY start.sh .
COPY templates/ ./templates/
COPY assets/ ./assets/

# Copy CSV file for training (required for cloud deployment)
COPY DBDData.csv ./

# Convert line endings and make startup script executable
RUN dos2unix start.sh && chmod +x start.sh

# Train models during Docker build (for fast startup in production)
# Models are pre-trained and included in the image
# This allows Render deployments to start quickly without retraining
# Local Docker containers can still retrain by setting SKIP_TRAINING=false or omitting it
# Set MODEL_OUTPUT_DIR to /app to ensure models are saved in the correct location
RUN echo "Training models during Docker build..." && \
    MODEL_OUTPUT_DIR=/app TRAINING_CSV=/app/DBDData.csv python train_model.py && \
    echo "Models trained successfully during build!" && \
    ls -lh /app/*.pth /app/*.pkl 2>/dev/null || echo "Warning: Model files not found in /app"

# Note: Models are pre-trained during Docker build for fast startup times
# - Render deployments use pre-trained models (SKIP_TRAINING=true) for fast startup (~30-60 seconds)
# - Local Docker containers can still retrain by not setting SKIP_TRAINING (default behavior)
# - Models will automatically retrain locally if data file has changed (detected via hash comparison)
# - DBDData.csv is included in the image, but can be overridden via volume mount or environment variable

# Expose port
EXPOSE 5000

# Health check (start period allows buffer for startup, though typically much faster with pre-trained models)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

# Run startup script (trains models if SKIP_TRAINING is not set, then starts Flask)
CMD ["/bin/bash", "./start.sh"]

