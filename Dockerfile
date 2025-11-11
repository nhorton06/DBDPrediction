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

# Note: Models are NOT pre-trained during build
# Models will be trained on every container startup to ensure they use the latest data
# This allows updating DBDData.csv and having models automatically retrain with new data
# DBDData.csv is included in the image, but can be overridden via volume mount or environment variable

# Expose port
EXPOSE 5000

# Health check (longer start period to allow for model training on startup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

# Run startup script (always trains models on startup, then starts Flask)
CMD ["/bin/bash", "./start.sh"]

