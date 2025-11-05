# Complete Guide: Training to Deployment

This guide walks you through the entire process from training your model to deploying it on the website.

## 📚 Overview

1. **Train the Model** - Use your Jupyter notebook to train the neural network
2. **Save the Model** - Export the trained model and scaler files
3. **Deploy Locally** - Run the Flask app on your computer
4. **Deploy with Docker** (Optional) - Containerize and deploy anywhere

---

## Step 1: Train Your Model

### 1.1 Open Your Notebook
```bash
jupyter notebook DBDCode.ipynb
```

### 1.2 Run All Training Cells
Execute all cells in order from the beginning. The important cells are:

- **Cell 5-7**: Load and preprocess data
- **Cell 10-16**: Prepare training/test sets and scale features
- **Cell 18-25**: Define and initialize the model
- **Cell 28**: **TRAIN THE MODEL** (this is the main training loop)

Make sure to run cell 28 completely. You should see output like:
```
Epoch 001: | Loss: 0.77220 | Accuracy: 41.429
Epoch 002: | Loss: 0.68984 | Accuracy: 56.429
...
Epoch 030: | Loss: 0.08935 | Accuracy: 100.000
```

### 1.3 Verify Training
After training completes (cell 28), you should have:
- `model` - Your trained PyTorch model
- `scaler` - Your fitted StandardScaler
- `X_train.shape[1]` - The input size (number of features)
- `Hidden_size` - The hidden layer size

---

## Step 2: Save Your Model

### 2.1 Create a New Cell
After cell 28 (training), add a new cell with this code:

```python
# Save the trained model and scaler for Flask API
import pickle
import os

# Ensure we're in the right directory
os.chdir(r'c:\Users\dacur\Desktop\DBD Project')

# Save model state
torch.save(model.state_dict(), 'dbd_model.pth')
print("✓ Model weights saved to dbd_model.pth")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to scaler.pkl")

# Save model architecture parameters
model_info = {
    'input_size': X_train.shape[1],
    'hidden_size': Hidden_size
}
with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Model info saved to model_info.pkl")

print("\n✅ All files saved successfully!")
print("\nFiles created:")
print("  - dbd_model.pth")
print("  - scaler.pkl")
print("  - model_info.pkl")
```

### 2.2 Run the Save Cell
Execute this cell. You should see confirmation messages that all files were saved.

### 2.3 Verify Files
Check that these 3 files exist in your project directory:
- `dbd_model.pth` (model weights)
- `scaler.pkl` (feature scaler)
- `model_info.pkl` (model architecture info)

**Important**: These files must be in the same directory as `app.py`!

---

## Step 3: Test Your Model (Optional)

Before deploying, test that the saved model works:

### 3.1 Test in Notebook
Add this cell to verify the model loads correctly:

```python
# Test loading the model
import pickle

# Load model info
with open('model_info.pkl', 'rb') as f:
    loaded_info = pickle.load(f)

# Initialize model with saved architecture
test_model = DBDModel(input_size=loaded_info['input_size'], 
                      hidden_size=loaded_info['hidden_size'])

# Load weights
test_model.load_state_dict(torch.load('dbd_model.pth'))
test_model.eval()

# Load scaler
with open('scaler.pkl', 'rb') as f:
    test_scaler = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"  Input size: {loaded_info['input_size']}")
print(f"  Hidden size: {loaded_info['hidden_size']}")

# Test prediction (using sample data)
sample = X_test[0:1]  # Take first test sample
sample_scaled = test_scaler.transform(sample)
sample_tensor = torch.FloatTensor(sample_scaled)

with torch.no_grad():
    prediction = test_model(sample_tensor)
    probability = torch.sigmoid(prediction).item()

print(f"\n✅ Test prediction: {probability*100:.2f}% escape chance")
```

---

## Step 4: Deploy Locally

### 4.1 Install Dependencies (if not already done)
```bash
python -m pip install -r requirements.txt
```

### 4.2 Verify Model Files Are Present
Make sure these files are in your project root:
```
DBD Project/
├── app.py              ✓
├── dbd_model.pth       ✓ (you just created this)
├── scaler.pkl          ✓ (you just created this)
├── model_info.pkl      ✓ (you just created this)
├── templates/
│   └── index.html      ✓
└── static/             ✓
```

### 4.3 Run the Flask App
```bash
python app.py
```

You should see:
```
Starting Flask server...
Server running on http://127.0.0.1:5000
Debug mode: ON
Model and scaler loaded successfully!
 * Running on http://127.0.0.1:5000
```

### 4.4 Access the Website
Open your browser and go to: **http://127.0.0.1:5000**

You should see the dark horror-themed interface!

### 4.5 Test a Prediction
1. Fill in all the form fields
2. Click "CALCULATE YOUR FATE"
3. You should see your escape probability

---

## Step 5: Deploy with Docker (Optional)

### 5.1 Build Docker Image
```bash
docker build -t dbd-predictor .
```

This creates a Docker image with all dependencies.

### 5.2 Run with Docker Compose (Easiest)
```bash
docker-compose up -d
```

This automatically:
- Builds the image
- Runs the container
- Mounts your model files
- Exposes port 5000

### 5.3 Run with Docker Directly
```bash
docker run -d -p 5000:5000 \
  -v "$(pwd)/dbd_model.pth:/app/dbd_model.pth:ro" \
  -v "$(pwd)/scaler.pkl:/app/scaler.pkl:ro" \
  -v "$(pwd)/model_info.pkl:/app/model_info.pkl:ro" \
  --name dbd-predictor dbd-predictor
```

On Windows PowerShell:
```powershell
docker run -d -p 5000:5000 `
  -v "${PWD}/dbd_model.pth:/app/dbd_model.pth:ro" `
  -v "${PWD}/scaler.pkl:/app/scaler.pkl:ro" `
  -v "${PWD}/model_info.pkl:/app/model_info.pkl:ro" `
  --name dbd-predictor dbd-predictor
```

### 5.4 Access Containerized App
Open: **http://localhost:5000**

### 5.5 View Logs
```bash
docker logs dbd-predictor
```

### 5.6 Stop Container
```bash
docker-compose down
# or
docker stop dbd-predictor
docker rm dbd-predictor
```

---

## Step 6: Update/Retrain Model

If you want to retrain with new data:

1. Update `DBDData.csv` with new data
2. Go back to Step 1 and retrain
3. Save again using Step 2
4. Restart the Flask app (it will automatically load the new model)

**No code changes needed** - just replace the `.pth` and `.pkl` files!

---

## 🔧 Troubleshooting

### Model Not Loading
**Error**: `Model not loaded. Please train and save the model first.`

**Solution**:
1. Check that `dbd_model.pth`, `scaler.pkl`, and `model_info.pkl` exist
2. Ensure they're in the same directory as `app.py`
3. Verify the files were created after training completed

### Model Architecture Mismatch
**Error**: `Error loading model: ...`

**Solution**:
- Make sure `Hidden_size` matches between training and deployment
- The model architecture in `app.py` must match your training notebook
- Retrain and save again

### Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Find what's using port 5000
netstat -ano | findstr :5000

# Or use a different port
FLASK_PORT=8080 python app.py
```

### Dependencies Missing
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
python -m pip install -r requirements.txt
```

### Docker Can't Find Model Files
**Error**: Model files not found in container

**Solution**:
- Ensure files exist before running Docker
- Check volume mount paths in `docker-compose.yml`
- Verify file permissions (should be readable)

---

## 📋 Quick Reference

### Training → Deployment Checklist
- [ ] Open notebook (`DBDCode.ipynb`)
- [ ] Run all cells including training (cell 28)
- [ ] Add save cell after training
- [ ] Execute save cell - verify 3 files created
- [ ] Install dependencies: `python -m pip install -r requirements.txt`
- [ ] Run app: `python app.py`
- [ ] Test in browser: http://127.0.0.1:5000
- [ ] (Optional) Build Docker: `docker build -t dbd-predictor .`
- [ ] (Optional) Run Docker: `docker-compose up -d`

### Files You Need
- ✓ `app.py` - Flask application
- ✓ `templates/index.html` - Web interface
- ✓ `requirements.txt` - Dependencies
- ✓ `dbd_model.pth` - **Created after training**
- ✓ `scaler.pkl` - **Created after training**
- ✓ `model_info.pkl` - **Created after training**

---

## 🚀 Production Deployment

For production (cloud deployment), see `DEPLOYMENT.md` for:
- GitHub setup
- Cloud platform deployment (Heroku, AWS, etc.)
- Docker Hub publishing
- CI/CD setup

---

## 📝 Notes

- **Model files are large**: `dbd_model.pth` can be 1-10MB
- **Training time**: Depending on your data, training may take a few minutes
- **No retraining needed**: The Flask app just loads the saved model - no training happens on the server
- **Same model, different environments**: Once saved, the same model files work locally, in Docker, or in the cloud

---

**Need help?** Check `INSTRUCTIONS.md` and `DEPLOYMENT.md` for more details!

