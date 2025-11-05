# Automatic Model Training on Container Startup

## Overview

The Docker container now automatically trains the model on the CSV file when it starts. Just mount your `DBDData.csv` file and the container will:
1. Train the model on startup
2. Save the model files
3. Start the Flask application

## How It Works

1. **`train_model.py`**: Standalone Python script that replicates the training pipeline from `DBDCode.ipynb`
2. **`start.sh`**: Startup script that runs training first, then starts Flask
3. **Dockerfile**: Updated to copy training script and use startup script
4. **docker-compose.yml**: Updated to mount CSV file instead of model files

## Usage

### Step 1: Ensure Your CSV is Ready

Make sure `DBDData.csv` is in your project root directory with the correct format:
- Columns: `Survivor Gender`, `Steam Player`, `Anonymous Mode`, `Prestige`, `Item`, `Map Area`, `Survivor BP`, `Killer BP`, `Result`
- Values: `Result` should be "Escape" or "Dead"

### Step 2: Build and Run

```bash
# Build the image
docker-compose build

# Start the container (will train on startup)
docker-compose up -d
```

### Step 3: Monitor Training

Watch the logs to see training progress:

```bash
docker-compose logs -f
```

You'll see output like:
```
==========================================
DBD Escape Prediction Container Startup
==========================================

Step 1: Training model on /app/DBDData.csv
----------------------------------------
[1/7] Loading data from /app/DBDData.csv...
   Loaded 544 rows
[2/7] Preprocessing data...
...
Epoch 001: | Loss: 0.77220 | Accuracy: 41.429%
...
Epoch 030: | Loss: 0.08617 | Accuracy: 100.000%
✅ Model training and saving completed successfully!

Step 2: Starting Flask application
----------------------------------------
Starting Flask server...
Server running on http://0.0.0.0:5000
```

### Step 4: Access the Application

Once training completes and Flask starts:
- Open `http://localhost:5000` in your browser

## Updating the CSV

To retrain with new data:

1. **Update `DBDData.csv`** with new rows
2. **Restart the container**:
   ```bash
   docker-compose restart
   ```
   The container will automatically retrain on the new CSV

## Important Notes

### Training Time
- Small datasets (< 1000 rows): ~30 seconds - 2 minutes
- Medium datasets (1000-5000 rows): ~2-5 minutes
- Large datasets (> 5000 rows): ~5-10 minutes

The health check waits 120 seconds before checking, but training may take longer for large datasets.

### Container Startup
- First startup: Trains model, then starts Flask
- Subsequent restarts: Retrains model (always uses latest CSV)
- Model files are generated inside the container at `/app/`

### CSV Requirements
- Must be named `DBDData.csv` (or set `TRAINING_CSV` environment variable)
- Must be in the project root (or mounted to `/app/DBDData.csv`)
- Must have the correct column structure (see above)

## Environment Variables

You can customize the training:

```yaml
environment:
  - TRAINING_CSV=/app/DBDData.csv  # Path to CSV file
  - MODEL_OUTPUT_DIR=/app          # Where to save model files
```

## Troubleshooting

### Training Fails
- Check CSV file exists and is readable
- Verify CSV format matches expected structure
- Check logs: `docker-compose logs -f`

### Container Takes Too Long
- Training takes time, especially on large datasets
- Check logs to see training progress
- Increase `start_period` in docker-compose.yml if needed

### Model Not Loading
- Check that training completed successfully (look for "✅ Model training and saving completed successfully!")
- Verify model files exist in container: `docker exec dbd-escape-predictor ls -la /app/*.pth /app/*.pkl`

## Benefits

✅ **Automatic retraining**: Update CSV and restart - no manual training needed
✅ **Always up-to-date**: Model trained on latest data every time
✅ **No manual steps**: Everything happens automatically
✅ **Consistent environment**: Same training process every time

## Manual Training (Optional)

If you want to train manually instead:

1. Comment out the training step in `start.sh`
2. Train locally using `train_model.py`:
   ```bash
   python train_model.py
   ```
3. Mount the model files as volumes in `docker-compose.yml`

