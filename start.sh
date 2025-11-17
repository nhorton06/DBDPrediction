#!/bin/bash
# Startup script that trains the model (if needed), then starts Flask
# Models are only retrained when the data file has changed
# Works both locally and in Docker containers

set -e  # Exit on error

echo "=========================================="
echo "DBD Escape Prediction Startup"
echo "=========================================="

# Detect if running in Docker or locally
if [ -d "/app" ] && [ -f "/app/app.py" ]; then
    # Running in Docker container
    WORK_DIR="/app"
    CSV_PATH="${TRAINING_CSV:-/app/DBDData.csv}"
    TRAIN_SCRIPT="train_model.py"
    APP_SCRIPT="app.py"
    echo "Running in Docker container"
else
    # Running locally
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CSV_PATH="${TRAINING_CSV:-${WORK_DIR}/DBDData.csv}"
    TRAIN_SCRIPT="src/train_model.py"
    APP_SCRIPT="run_local.py"
    echo "Running locally"
    cd "$WORK_DIR"
fi

echo "Working directory: $WORK_DIR"
echo ""

# Check if CSV file exists
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: Training CSV not found at $CSV_PATH"
    echo "Please ensure DBDData.csv is in the project root directory"
    exit 1
fi

# Check if training should be skipped (for Render/production deployments)
if [ "${SKIP_TRAINING:-false}" = "true" ]; then
    echo "Step 1: Skipping model training (SKIP_TRAINING=true)"
    echo "----------------------------------------"
    echo "Using pre-trained models"
    echo ""
else
    echo "Step 1: Checking if model training is needed"
    echo "----------------------------------------"
    
    # Set environment variables for training
    export TRAINING_CSV="$CSV_PATH"
    if [ -z "$MODEL_OUTPUT_DIR" ]; then
        export MODEL_OUTPUT_DIR="$WORK_DIR"
    fi
    
    python "$TRAIN_SCRIPT"

    if [ $? -ne 0 ]; then
        echo "ERROR: Model training failed"
        exit 1
    fi
    echo ""
fi

echo "Step 2: Starting Flask application"
echo "----------------------------------------"
exec python "$APP_SCRIPT"

