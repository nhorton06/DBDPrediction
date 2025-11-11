# Dead by Daylight Escape Prediction - Case Study

## 1) Executive Summary

### Problem

Dead by Daylight (DBD) players, especially ones like myself who have dedicated thousands of hours to the game, want to understand a survivor's chance of escaping a match based on various in-game factors. Currently, I basically just have to rely on intuition and my personal experience to estimate escape probability. I can also sometimes look at a survivor's hour count in-game, but this is not always available as it is entirely up to the player to make this public or not. This project addresses the need for a data-driven prediction tool that helps potentially make more informed decisions during pre-game lobby queue and understand how certain factors (survivor characteristics, map size, items, prestige level) may or may not influence escape rates.

**Target Users**: Dead by Daylight players, gaming analysts, and researchers interested in game mechanics analysis

### Solution

This project is a machine learning web application that predicts a survivor's probability of escaping in Dead by Daylight using a PyTorch neural network. The system accepts inputs such as survivor characteristics, prestige level, items and add-ons, map information, perks (exhaustion, chase, and other perks), and player statistics (bloodpoints), then outputs a probability score between 0-100% indicating the likelihood of escape or sacrifice. The application is delivered as a Flask web app with a user-friendly, categorized interface deployed through Docker, making it accessible to both technical and non-technical users. The model automatically trains on startup using the provided CSV data, ensuring it's always up-to-date with the latest gameplay patterns.

## 2) System Overview

### Course Concept(s)

- **Flask API**: Putting my neural network model into a Flask app for deployment into an html file to be interactive for the user
- **Containerization**: Docker for consistent deployment and environment management

### Architecture Diagram

The system follows a request-response flow with multiple endpoints:

**Prediction Flow:**
1. User submits form data via web interface
2. Flask receives and validates input (including perk count validation)
3. Features are preprocessed and scaled
4. PyTorch model performs inference
5. Result (escape probability) is returned and displayed
6. Optionally, feature importance is calculated and displayed

**Optimization Flow:**
1. User clicks "Optimize" button with current form values
2. Flask receives current configuration
3. Optimization algorithm searches for best variable combinations (respecting 4-perk limit)
4. Optimal configuration is returned with escape chance improvement
5. User can apply optimized values to the form

**Feature Importance Flow:**
1. User checks "Show feature influence" and submits prediction
2. Flask calculates gradients using backpropagation
3. Feature importance (magnitude and direction) is computed
4. Results are displayed showing which features increase/decrease escape chance

*Note: A visual architecture diagram can be added to `/assets/architecture.png` if needed*

**System Flow:**
1. User inputs game parameters via web interface or API
2. Flask application receives and validates input (perk count, data types, ranges)
3. Features are preprocessed and scaled using saved scaler
4. PyTorch model performs inference
5. Optional: Feature importance calculated via gradient analysis
6. Result is formatted as probability percentage and returned to user
7. Optional: Optimization endpoint finds best configuration to maximize escape chance

### Data/Models/Services

**Training Data:**
- **Source**: `DBDData.csv` (gameplay data)
- **Size**: Varies by dataset (typically thousands of gameplay records)
- **Format**: CSV with columns for survivor attributes, game settings, and outcomes
- **License**: Project-specific data (not publicly distributed)

**Model:**
- **Type**: PyTorch Neural Network (2 hidden layers, batch normalization, dropout)
- **Dual Models**: Two models are trained and available:
  - **With Bloodpoints**: Includes survivor BP and killer BP as features (31 total features)
  - **Without Bloodpoints**: Excludes bloodpoint features (29 total features)
- **Input Features** (With BP model):
  - Player characteristics (gender, steam player, anonymous mode, prestige)
  - Items and equipment (item type, powerful add-ons)
  - Map information (map type, map area)
  - Perks (exhaustion perks, chase perks count, decisive strike, unbreakable, off the record, adrenaline)
  - Bloodpoints (survivor BP, killer BP)
- **Input Features** (Without BP model): Same as above, excluding bloodpoint features
- **Output**: Binary classification probability (escape vs. sacrifice)
- **Training**: Automatically trains both models on container startup with early stopping (max 100 epochs, patience 5)
- **Format**: Saved as `dbd_model_with_bp.pth` / `dbd_model_no_bp.pth` (PyTorch state dict), `scaler_with_bp.pkl` / `scaler_no_bp.pkl` (scikit-learn StandardScaler), `model_info_with_bp.pkl` / `model_info_no_bp.pkl` (architecture metadata)
- **Size**: Model weights ~few MB each, total artifacts ~20-100 MB

**Services:**
- **Web Interface**: Flask web server (port 5000) with toggle switch to select between models
- **API Endpoints**:
  - **POST `/predict`**: Main prediction endpoint (includes optional `include_importance` flag for feature influence)
  - **POST `/optimize`**: Optimization endpoint to find best variable configuration (respects 4-perk limit)
  - **GET `/health`**: Health check for container monitoring (reports status of both models)
- **Features**:
  - Real-time escape probability prediction
  - Feature importance analysis (gradient-based, shows direction and magnitude)
  - Configuration optimization (finds best settings to maximize escape chance)
  - Perk validation (enforces 4-perk limit for survivors)
  - Dual model support (with/without bloodpoints)

## 3) How to Run (Local)

### **Important Note**: Setting up the container and running it for the first time can take a pretty large amount of time (upwards of 20 minutes). Please be patient as this project uses machine learning and neural network modules.

### Prerequisites

- Docker and Docker Compose installed (for Docker method)
- OR Python 3.8+ installed (for local Python method)
- `DBDData.csv` file in the project root (will be used for training)

### Local Python Development (Without Docker)

If you want to run the application directly on your machine without Docker:

**1. Install Python Dependencies:**

```bash
# Install required packages
pip install -r requirements.txt
```

**2. Train Models (First Time Only):**

Before running the app, you need to train the models. The models will be saved in the project root directory:

```bash
# Train both models (with and without bloodpoints)
python src/train_model.py

# This will create:
# - dbd_model_with_bp.pth
# - dbd_model_no_bp.pth
# - scaler_with_bp.pkl
# - scaler_no_bp.pkl
# - model_info_with_bp.pkl
# - model_info_no_bp.pkl
```

**Note**: Model training can take 2-5 minutes depending on your system. The models are saved to the current directory (project root) by default.

**3. Run the Flask Application:**

Use the provided `run_local.py` script which handles path setup correctly:

```bash
python run_local.py
```

The app will start on `http://127.0.0.1:5000` (or `http://localhost:5000`).

**Alternative: Run Directly (Not Recommended):**

If you prefer to run directly, you must be in the project root directory:

```bash
# Make sure you're in the project root
cd /path/to/DBDPrediction

# Set environment variable
export FLASK_APP=src/app.py  # On Windows: set FLASK_APP=src/app.py

# Run Flask
python -m flask run --debug
```

**Important Notes for Local Development:**

- Models must be trained before running the app (use `python src/train_model.py`)
- Models are saved to the project root directory (`.`), not `/app/` like in Docker
- The `run_local.py` script automatically sets up the correct paths for templates and models
- If you see "Model not loaded" errors, make sure you've trained the models first
- The app looks for models in: current directory, project root, `/app/`, and `project_root/app/`

### Docker (Single Command)

```bash
# Build the image
docker build -t dbd-predictor:latest .

# Run the container (mounts DBDData.csv for training)
docker run --rm -p 5000:5000 \
  -v $(pwd)/DBDData.csv:/app/DBDData.csv:ro \
  -e FLASK_ENV=production \
  -e FLASK_APP=app.py \
  -e TRAINING_CSV=/app/DBDData.csv \
  -e MODEL_OUTPUT_DIR=/app \
  dbd-predictor:latest

# Health check (in another terminal)
curl http://localhost:5000/health
```

### Docker Hub (Pre-built Image)

```bash
# Pull the pre-built image from Docker Hub
docker pull h2x0/dbd-predictor:latest

# Run the container
docker run --rm -p 5000:5000 \
  -v $(pwd)/DBDData.csv:/app/DBDData.csv:ro \
  -e FLASK_ENV=production \
  -e FLASK_APP=app.py \
  -e TRAINING_CSV=/app/DBDData.csv \
  -e MODEL_OUTPUT_DIR=/app \
  h2x0/dbd-predictor:latest
```

**Note**: The Docker Hub image includes `DBDData.csv` in the image, so the volume mount is optional (useful for overriding with local data).

### Docker Compose (Recommended)

```bash
# Build and start the container (one command startup)
docker-compose up -d

# The container will:
# 1. Automatically train the model on DBDData.csv
# 2. Start the Flask web server
# 3. Be accessible at http://localhost:5000

# View logs (including training progress)
docker-compose logs -f

# Health check
curl http://localhost:5000/health

# Stop the container
docker-compose down

# Rebuild after code changes
docker-compose down
docker-compose build
docker-compose up -d
```

**Note**: Models are trained on every container startup (takes 2-5 minutes) to ensure they use the latest data. This allows you to update `DBDData.csv` and have the models automatically retrain with the new data. The web interface includes a toggle switch to select between the "With Bloodpoints" and "Without Bloodpoints" models.

**For Local Python Development**: Models must be trained manually using `python src/train_model.py` before running the app. See the "Local Python Development" section above for details.

### Updating Docker Hub Image

To update the Docker Hub image with the latest changes:

```bash
# Build the image
docker build -t h2x0/dbd-predictor:latest .

# Push to Docker Hub
docker push h2x0/dbd-predictor:latest
```

**Note**: The image is available at `h2x0/dbd-predictor:latest` on Docker Hub.

### Cloud Deployment (Render)

The project includes a `render.yaml` configuration file for easy deployment to Render:

**1. Deploy to Render:**

- Connect your GitHub repository to Render
- Render will automatically detect `render.yaml` and deploy the service
- The service will train models on startup (takes 2-5 minutes)

**2. Updating Data on Render:**

To update the training data (`DBDData.csv`) on Render:

- **Option A**: Update `DBDData.csv` in your repository and push to GitHub. Render will automatically redeploy, and models will retrain with the new data.
- **Option B**: Use Render's environment variables or file system to update the CSV file, then trigger a manual redeploy.

**Important Notes for Render Deployment:**

- Models train on every deployment/startup (2-5 minutes)
- Health check start period is set to 300 seconds (5 minutes) to allow for training
- Ensure your Render plan has sufficient resources for model training
- The `DBDData.csv` file is included in the Docker image, but you can override it via environment variables or file mounts

## 4) Design Decisions

### Why This Concept?

**Docker**: Essential for ensuring consistent deployment across different environments. The container includes all dependencies (PyTorch, Flask, scikit-learn) and eliminates "works on my machine" issues. The container automatically trains the model on startup, ensuring it's always up-to-date with the latest data.

**Flask**: Chosen for its simplicity and lightweight nature. The application only needs basic routing and JSON handling, making Flask more appropriate than heavier frameworks like Django. The REST API design allows for both web interface and programmatic access.

### Tradeoffs

**Performance vs. Model Size**: 
- **Decision**: CPU-only PyTorch for smaller image size and faster builds
- **Tradeoff**: Smaller image size and faster builds, but no GPU acceleration
- **Impact**: Build times are quick (~30 seconds), and inference is fast enough for web use (~5-10 ms per prediction)

**Dual Model Approach**:
- **Decision**: Train two separate models (with/without bloodpoints) to allow users to choose
- **Tradeoff**: Doubles training time and storage, but provides flexibility
- **Rationale**: Bloodpoints may be overly influential on predictions; users can choose based on their preference

**Complexity vs. Maintainability**:
- **Decision**: Custom neural network architecture vs. simpler models
- **Tradeoff**: More complex model requires careful architecture matching between training and inference
- **Mitigation**: Model metadata (`model_info.pkl`) stores architecture parameters to ensure compatibility

**Development vs. Production**:
- **Decision**: Automatic model training on startup vs. pre-trained models
- **Tradeoff**: Training on startup ensures fresh models but increases startup time
- **Rationale**: Automatic training ensures models are always up-to-date with the latest data
- **Early Stopping**: Implemented to prevent overfitting and improve generalization as dataset grows (max 100 epochs, patience 5)

### Security/Privacy

**Secrets Management**: 
- No API keys or sensitive credentials required
- Model files contain no user data (pre-trained weights only)
- Environment variables used for configuration (FLASK_HOST, FLASK_PORT, FLASK_DEBUG)

**Input Validation**:
- Flask automatically handles JSON parsing errors
- Input ranges validated on frontend (e.g., prestige 0-100)
- Type checking in prediction endpoint (float conversion with error handling)
- **Known Limitation**: No explicit rate limiting (acceptable for single-user/local deployment)

**PII Handling**:
- No personally identifiable information collected or stored
- All inputs are game statistics (gender, items, points) - no real names or identifiers
- No user session tracking or logging of predictions

**Data Privacy**:
- Training data (`DBDData.csv`) excluded from repository via `.gitignore`
- Model files excluded to prevent accidental data leakage
- Users must train their own models or provide their own model files

**Network Security**:
- HTTP-only deployment (no HTTPS enforcement in current setup)
- No authentication/authorization mechanisms
- Suitable for local/development use; production deployment should add HTTPS and access controls

### Ethics

**Algorithmic Fairness**:
- Model predictions are based solely on in-game statistics and do not discriminate based on real-world characteristics
- No demographic data (race, age, location) is collected or used in predictions
- Gender field refers to in-game character gender, not player gender

**Transparency**:
- Model architecture and training process are documented in the codebase
- Feature importance analysis allows users to understand which variables influence predictions
- Optimization feature shows recommended changes, allowing users to make informed decisions

**Data Collection & Usage**:
- No user data is collected or stored during predictions
- Training data consists of anonymized gameplay statistics only
- No tracking, analytics, or user behavior monitoring

**Responsible AI**:
- Model predictions are probabilistic estimates, not deterministic outcomes
- Users should understand predictions are based on historical data patterns
- No guarantee of accuracy; predictions are for informational purposes only

### Operations & Risk Management

**Operational Risks**:
1. **Model Training Failures**: If training fails on startup, container will not serve requests
   - **Mitigation**: Health check endpoint reports model status; container will fail health checks if models aren't loaded
   - **Recovery**: Manual container restart or redeployment
2. **Resource Exhaustion**: Model training on startup consumes significant CPU/memory
   - **Mitigation**: Health check start period set to 300s (5 minutes) to allow training time
   - **Risk**: On resource-constrained systems (e.g., Render free tier with 512MB RAM), training may fail or timeout
3. **Data Availability**: Missing or corrupted `DBDData.csv` will cause training to fail
   - **Mitigation**: CSV file included in Docker image; can be overridden via volume mount
4. **Single Point of Failure**: Single-container deployment has no redundancy
   - **Mitigation**: Stateless design allows easy redeployment; no data loss risk
5. **No Backup/Recovery**: Model files are generated at runtime and not persisted
   - **Impact**: Container restart triggers retraining (intentional design for fresh models)

**Monitoring & Observability**:
- **Health Checks**: `/health` endpoint reports model loading status for both models
- **Logging**: Flask development server logs all requests to stdout (captured by Docker)
- **Limitations**: 
  - No structured logging (JSON format)
  - No metrics collection (request rates, latency, error rates)
  - No alerting mechanisms
  - No distributed tracing
- **Future Improvements**: Could add Prometheus metrics, structured logging (JSON), and error tracking (Sentry)

**Scaling Considerations**:
- Current design: Single-container deployment
- **Limitation**: No horizontal scaling capability (would require load balancer, shared state)
- **Future**: Could add Redis for session management, gunicorn for multi-worker support

**Resource Footprint**:
- **Container Image**: ~500 MB - 1 GB (Python 3.11 slim base with dependencies)
- **Memory Usage**:
  - Runtime (inference only): ~500 MB - 1 GB
  - During training (startup): ~1-2 GB peak (both models training simultaneously)
  - Model storage: ~20-100 MB for both models and scalers
- **CPU Usage**:
  - Inference: Minimal (~5-10 ms per prediction, single-threaded)
  - Training: High CPU usage during startup (2-5 minutes, both models)
- **Disk I/O**: Minimal after startup (models loaded into memory)
- **GPU**: Not currently used (CPU-only PyTorch)
- **Network**: Minimal bandwidth (small JSON requests/responses)

**Known Limitations**:
1. Both models train on every container startup (adds ~2-5 minutes to startup time) - this is intentional to ensure models use the latest data
2. No database persistence (stateless predictions only)
3. Single-threaded Flask server (not production-grade for high traffic)
4. No authentication/authorization (suitable for local use only)
5. Training data can be provided via volume mount (`DBDData.csv`) or is included in Docker Hub image
6. **Note**: To update models with new data, simply update `DBDData.csv` and redeploy - models will automatically retrain on startup

## 5) Results & Evaluation

### Screenshots

![Web Interface](assets/web-interface.png)
*Main prediction interface with form inputs*

![Prediction Result](assets/prediction-result.png)
*Example prediction showing 67% escape probability*

*Note: Place screenshots in `/assets/` directory*

### Sample Outputs

**API Response Example:**
```json
{
  "escape_chance": 67.45,
  "will_escape": true,
  "probability": 0.6745,
  "model_type": "with_bp"
}
```

**API Request Example:**
```json
{
  "model_type": "with_bp",
  "survivor_gender": "Female",
  "steam_player": "Yes",
  "anonymous_mode": "No",
  "item": "Flashlight",
  "prestige": 5,
  "map_type": "Indoor",
  "map_area": 8500,
  "powerful_add_ons": "Yes",
  "exhaustion_perk": "Sprint Burst",
  "chase_perks": 2,
  "decisive_strike": "Yes",
  "unbreakable": "No",
  "off_the_record": "No",
  "adrenaline": "Yes",
  "survivor_bp": 25000,
  "killer_bp": 18000
}
```

**Note**: When using `model_type: "no_bp"`, omit `survivor_bp` and `killer_bp` from the request.

**Web Interface:**
- Model selection toggle switch (With Bloodpoints / Without Bloodpoints)
- Categorized form sections (Player Characteristics, Items, Map Information, Perks, Match Bloodpoints)
- Bloodpoints section dynamically shows/hides based on selected model
- Icon-enhanced inputs for better visual recognition
- Displays escape probability as percentage
- Color-coded results (green for escape, red for sacrifice)
- Outcome images (escaped.png / sacrificed.png)
- Visual feedback with animations and glow effects

### Performance Notes

**Inference Speed**:
- CPU: ~5-10 ms per prediction
- GPU (if available): ~1-2 ms per prediction
- Network latency: Minimal (local deployment)

**Resource Usage**:
- Container startup: ~2-5 minutes (includes training both models on every startup)
- Training time: ~30-90 seconds per model (varies with early stopping, max 100 epochs)
- Memory footprint: ~500 MB base + model size (both models loaded in memory)
- Disk I/O: Minimal (models loaded once after training)
- **Note**: Models train on every startup to ensure they use the latest data from `DBDData.csv`

### Validation/Tests

**Model Validation**:
- Trained model evaluated on test set (20% holdout)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Typical accuracy: ~99% on training data
- Model training includes gradient clipping and NaN handling for stability

**API Testing**:
- Health check endpoint verified
- Input validation tested with edge cases (invalid ranges, missing fields)
- Error handling verified for malformed requests

**Integration Testing**:
- Docker container build verified
- Automatic model training on startup verified
- End-to-end workflow: form submission → prediction → display
- Unit tests available in `tests/` directory

## 6) What's Next

### Project Structure

```
DBDPrediction/
├── src/                    # Application source code
│   ├── app.py             # Flask web application (loads both models, handles predictions)
│   ├── train_model.py     # Model training script (trains both with_bp and no_bp models)
│   └── save_model.py      # Model saving utility
├── tests/                  # Unit and smoke tests
│   ├── test_api.py        # API endpoint tests
│   └── test_model.py      # Model architecture tests
├── templates/              # HTML templates
│   └── index.html         # Main web interface (includes model toggle switch)
├── assets/                 # Images and static assets
│   ├── dbd-logo.png       # Dead by Daylight logo
│   ├── icon.png           # Favicon
│   ├── escaped.png        # Escape outcome image
│   ├── sacrificed.png     # Sacrifice outcome image
│   ├── *.png              # Various UI icons (steam, anonymous, prestige, items, perks, etc.)
├── DBDData.csv            # Training data (included in Docker image, not in repo)
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── render.yaml           # Render cloud deployment configuration
├── requirements.txt       # Python dependencies
├── run_local.py          # Local development runner (sets up paths, runs Flask)
├── start.sh              # Container startup script (trains models, then starts Flask)
├── README.md             # Project documentation
└── CREDITS.md            # Attribution for game assets and third-party resources
```

**Generated Model Files** (created at runtime, not in repo):
- `dbd_model_with_bp.pth` - PyTorch model weights (with bloodpoints)
- `dbd_model_no_bp.pth` - PyTorch model weights (without bloodpoints)
- `scaler_with_bp.pkl` - Feature scaler (with bloodpoints)
- `scaler_no_bp.pkl` - Feature scaler (without bloodpoints)
- `model_info_with_bp.pkl` - Model metadata (with bloodpoints)
- `model_info_no_bp.pkl` - Model metadata (without bloodpoints)

### Planned Improvements

   - Collect more data to improve model accuracy
   - Update data as game patches and updates are released
   - Add more sophisticated feature engineering
   - Implement model versioning
   - Export weights for analyzing potential improvements based off heavier weights (more impactful variables)
   - Cloud deployment

### Recent Updates

   - **Feature Importance Analysis**: Added gradient-based feature influence calculation showing which variables increase/decrease escape chance
   - **Configuration Optimization**: Added optimization endpoint that finds the best variable combination to maximize escape chance (respects 4-perk limit)
   - **Perk Validation**: Enforced 4-perk limit validation on both frontend and backend
   - **Dual Model Support**: Added toggle to switch between models with/without bloodpoint features
   - **Early Stopping**: Implemented early stopping (max 100 epochs, patience 5) for better generalization
   - **Docker Hub Deployment**: Image available at `h2x0/dbd-predictor:latest` (models train on startup)
   - **Cloud Deployment**: `render.yaml` configuration included for easy Render deployment
   - **Automatic Model Retraining**: Models automatically retrain on every deployment to use latest data
   - Added comprehensive perk support (exhaustion, chase, other perks)
   - Added map type and powerful add-ons features
   - Organized UI into categorized sections with model selection toggle
   - Enhanced interface with icons for better UX
   - Simplified outcome display
   - Added unit tests
   - Improved code organization (src/ folder structure)
   - Added local development runner (`run_local.py`) for easier local hosting

### Stretch Features

   - It would be nice to be able to eventually collaborate with a group like NightLight to actively analyze auto-screenshots from games from the community rather than just my killer games to increase accuracy


## 7) Links (Required)

**GitHub Repo**: https://github.com/nhorton06/DBDPrediction

**Public Cloud App**: https://dbd-calculator.onrender.com/

**Credits & Attribution**: See [CREDITS.md](CREDITS.md) for attribution of game assets and third-party resources.