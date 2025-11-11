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

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Pre-train models during build to avoid startup delays
# This ensures models are ready immediately when container starts
RUN echo "Training models during Docker build..." && \
    TRAINING_CSV=/app/DBDData.csv MODEL_OUTPUT_DIR=/app python train_model.py && \
    echo "Models trained successfully!"

# Note: Models are now pre-trained in the image for faster startup
# DBDData.csv is included in the image, but can be overridden via volume mount or environment variable
# If models don't exist at startup, start.sh will train them (for local development with custom CSV)

# Expose port
EXPOSE 5000

# Health check (reduced start period since models are pre-trained)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

# Run startup script (will skip training if models exist, then starts Flask)
CMD ["/bin/bash", "./start.sh"]

