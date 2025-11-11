#!/bin/bash
# Startup script that trains the model if needed, then starts Flask

set -e  # Exit on error

echo "=========================================="
echo "DBD Escape Prediction Container Startup"
echo "=========================================="

# Check if models already exist (pre-trained during build)
MODEL_WITH_BP="/app/dbd_model_with_bp.pth"
MODEL_NO_BP="/app/dbd_model_no_bp.pth"

if [ -f "$MODEL_WITH_BP" ] && [ -f "$MODEL_NO_BP" ]; then
    echo "Pre-trained models found. Skipping training."
    echo "Models will be loaded from: /app/"
else
    echo "Models not found. Training models now..."
    echo ""
    
    # Check if CSV file exists
    CSV_PATH="${TRAINING_CSV:-/app/DBDData.csv}"
    if [ ! -f "$CSV_PATH" ]; then
        echo "ERROR: Training CSV not found at $CSV_PATH"
        echo "Please mount DBDData.csv as a volume or ensure it's in the image"
        exit 1
    fi

    echo "Step 1: Training model on $CSV_PATH"
    echo "----------------------------------------"
    python train_model.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Model training failed"
        exit 1
    fi
    echo ""
fi

echo "Step 2: Starting Flask application"
echo "----------------------------------------"
exec python app.py

