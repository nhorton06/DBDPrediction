#!/bin/bash
# Startup script that trains the model, then starts Flask
# Models are always trained on startup to ensure they use the latest data

set -e  # Exit on error

echo "=========================================="
echo "DBD Escape Prediction Container Startup"
echo "=========================================="

# Always train models on startup to use the latest data
echo "Training models with latest data..."
echo ""

# Check if CSV file exists
CSV_PATH="${TRAINING_CSV:-/app/DBDData.csv}"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: Training CSV not found at $CSV_PATH"
    echo "Please mount DBDData.csv as a volume or ensure it's in the image"
    exit 1
fi

echo "Step 1: Training models on $CSV_PATH"
echo "----------------------------------------"
python train_model.py

if [ $? -ne 0 ]; then
    echo "ERROR: Model training failed"
    exit 1
fi
echo ""

echo "Step 2: Starting Flask application"
echo "----------------------------------------"
exec python app.py

