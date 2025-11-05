# 🔮 DBD Escape Prediction API

A machine learning web application that predicts a survivor's chance of escaping in Dead by Daylight (DBD) using a PyTorch neural network. This project includes a Flask API with a user-friendly web interface and is containerized with Docker for easy deployment.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 🎯 Features

- **Interactive Web Interface**: Easy-to-use form with dropdowns for all input fields
- **REST API**: JSON API endpoint for programmatic access
- **Machine Learning Model**: PyTorch neural network trained on DBD gameplay data
- **Docker Support**: Fully containerized for consistent deployment across environments
- **Beautiful UI**: Modern, responsive design with gradient styling

## 📋 Prerequisites

- Python 3.11+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- Trained model files (see Setup section)

## 🚀 Quick Start with Docker (Recommended)

### Option 1: Using Docker Compose

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "DBD Project"
   ```

2. **Add your trained model files**
   - Copy `dbd_model.pth`, `scaler.pkl`, and `model_info.pkl` to the project root
   - These files are created after training the model (see Model Training section)

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Open your browser to `http://localhost:5000`

5. **Stop the container**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the Docker image**
   ```bash
   docker build -t dbd-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -d -p 5000:5000 \
     -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
     -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
     -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
     --name dbd-predictor dbd-predictor
   ```

3. **Access the application**
   - Open your browser to `http://localhost:5000`

## 🏗️ Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "DBD Project"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train and save your model**
   - Open `DBDCode.ipynb` in Jupyter
   - Run all cells to train the model
   - After training (after cell 28), run this code in a new cell:
   ```python
   import pickle
   
   torch.save(model.state_dict(), 'dbd_model.pth')
   with open('scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)
   model_info = {
       'input_size': X_train.shape[1],
       'hidden_size': Hidden_size
   }
   with open('model_info.pkl', 'wb') as f:
       pickle.dump(model_info, f)
   print("Model saved successfully!")
   ```

5. **Run the Flask application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
DBD Project/
├── app.py                 # Flask application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Files to exclude from Docker build
├── .gitignore           # Git ignore rules
├── DBDCode.ipynb        # Jupyter notebook for model training
├── DBDData.csv          # Training dataset (optional)
├── dbd_model.pth        # Trained model weights (create after training)
├── scaler.pkl           # Feature scaler (create after training)
└── model_info.pkl       # Model architecture info (create after training)
```

## 🎮 Using the Web Interface

1. Fill in the form fields:
   - **Survivor Gender**: Female or Male
   - **Steam Player**: Yes or No
   - **Anonymous Mode**: Yes or No
   - **Item Brought**: Select one item (Firecracker, Flashlight, Key, Map, Medkit, Toolbox, or None)
   - **Prestige**: Enter a number (0-100)
   - **Map Area**: Enter a number (in $m^2$)
   - **Survivor BP**: Enter the survivor's bloodpoints
   - **Killer BP**: Enter the killer's bloodpoints

2. Click **"Predict Escape Chance"**

3. View the prediction:
   - Whether you will escape (Yes/No)
   - The escape probability percentage (confidence)

## 🔌 API Endpoint

### POST `/predict`

Predict escape probability from JSON data.

**Request Body:**
```json
{
    "survivor_gender": "F",
    "steam_player": "Yes",
    "anonymous_mode": "No",
    "item": "Medkit",
    "prestige": 5,
    "map_area": 8000,
    "survivor_bp": 20000,
    "killer_bp": 25000
}
```

**Response:**
```json
{
    "escape_chance": 45.23,
    "will_escape": false,
    "probability": 0.4523
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "survivor_gender": "F",
    "steam_player": "Yes",
    "anonymous_mode": "No",
    "item": "Medkit",
    "prestige": 5,
    "map_area": 8000,
    "survivor_bp": 20000,
    "killer_bp": 25000
  }'
```

## 🐳 Docker Deployment

### Building the Image

```bash
docker build -t dbd-predictor .
```

### Running the Container

```bash
docker run -d -p 5000:5000 \
  -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
  -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
  -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
  --name dbd-predictor dbd-predictor
```

### Environment Variables

You can customize the Flask app behavior with environment variables:

```bash
docker run -d -p 5000:5000 \
  -e FLASK_HOST=0.0.0.0 \
  -e FLASK_PORT=5000 \
  -e FLASK_DEBUG=False \
  -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
  -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
  -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
  dbd-predictor
```

### Docker Compose

The `docker-compose.yml` file makes it easier to manage the container:

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🌐 Deploying to Cloud Platforms

### Heroku

1. Create a `Procfile`:
   ```
   web: python app.py
   ```

2. Update `app.py` to use the PORT environment variable:
   ```python
   port = int(os.getenv('PORT', 5000))
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### AWS, Google Cloud, Azure

Use the Docker container with their container services (ECS, Cloud Run, Container Instances).

## 🧪 Model Training

The model is trained using a PyTorch neural network. See `DBDCode.ipynb` for the complete training pipeline:

- Data preprocessing and feature engineering
- Train/test split
- Feature standardization
- Neural network architecture (2 hidden layers with batch normalization)
- Training with Adam optimizer
- Model evaluation

## 🔧 Troubleshooting

### Model Not Loading
- Ensure `dbd_model.pth`, `scaler.pkl`, and `model_info.pkl` exist in the project root
- Check file permissions if using Docker volumes
- Verify the files were created correctly during model training

### Port Already in Use
- Change the port in `app.py` or use the `FLASK_PORT` environment variable
- Update Docker port mapping: `-p 8080:5000` (uses port 8080 on host)

### Docker Build Fails
- Ensure Docker is running
- Check internet connection (needs to download base image)
- Verify `requirements.txt` has valid package names

### Container Can't Access Model Files
- Check volume mount paths in `docker-compose.yml`
- Ensure files exist before starting the container
- Verify file permissions

## 📝 License

This project is for educational purposes. Please respect Dead by Daylight's terms of service.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Remember to train your model and save the model files before running the application. The model files (`dbd_model.pth`, `scaler.pkl`, `model_info.pkl`) are not included in the repository and must be generated through training.

