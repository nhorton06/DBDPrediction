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

The system follows a simple request-response flow:
1. User submits form data via web interface
2. Flask receives and validates input
3. Features are preprocessed and scaled
4. PyTorch model performs inference
5. Result (escape probability) is returned and displayed

*Note: A visual architecture diagram can be added to `/assets/architecture.png` if needed*

**System Flow:**
1. User inputs game parameters via web interface or API
2. Flask application receives and validates input
3. Features are preprocessed and scaled using saved scaler
4. PyTorch model performs inference
5. Result is formatted as probability percentage and returned to user

### Data/Models/Services

**Training Data:**
- **Source**: `DBDData.csv` (gameplay data)
- **Size**: Varies by dataset (typically thousands of gameplay records)
- **Format**: CSV with columns for survivor attributes, game settings, and outcomes
- **License**: Project-specific data (not publicly distributed)

**Model:**
- **Type**: PyTorch Neural Network (2 hidden layers, batch normalization, dropout)
- **Input Features**: 31 features including:
  - Player characteristics (gender, steam player, anonymous mode, prestige)
  - Items and equipment (item type, powerful add-ons)
  - Map information (map type, map area)
  - Perks (exhaustion perks, chase perks count, decisive strike, unbreakable, off the record, adrenaline)
  - Bloodpoints (survivor BP, killer BP)
- **Output**: Binary classification probability (escape vs. sacrifice)
- **Training**: Automatically trains on container startup (15 epochs)
- **Format**: Saved as `dbd_model.pth` (PyTorch state dict), `scaler.pkl` (scikit-learn StandardScaler), `model_info.pkl` (architecture metadata)
- **Size**: Model weights ~few MB, total artifacts ~10-50 MB

**Services:**
- **Web Interface**: Flask web server (port 5000)
- **API Endpoint**: POST `/predict` for programmatic access
- **Health Check**: GET `/health` for container health monitoring

## 3) How to Run (Local)

### **Important Note**: Setting up the container and running it for the first time can take a pretty large amount of time (upwards of 20 minutes). Please be patient as this project uses machine learning and neural network modules.

### Prerequisites

- Docker and Docker Compose installed
- `DBDData.csv` file in the project root (will be used for training)

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

### Docker Compose (Recommended)

```bash
# Build and start the container
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

**Note**: The model trains automatically on startup, so the first request may take 1-2 minutes while training completes. Subsequent requests are instant.

## 4) Design Decisions

### Why This Concept?

**Docker**: Essential for ensuring consistent deployment across different environments. The container includes all dependencies (PyTorch, Flask, scikit-learn) and eliminates "works on my machine" issues. The container automatically trains the model on startup, ensuring it's always up-to-date with the latest data.

**Flask**: Chosen for its simplicity and lightweight nature. The application only needs basic routing and JSON handling, making Flask more appropriate than heavier frameworks like Django. The REST API design allows for both web interface and programmatic access.

### Tradeoffs

**Performance vs. Model Size**: 
- **Decision**: CPU-only PyTorch for smaller image size and faster builds
- **Tradeoff**: Smaller image size and faster builds, but no GPU acceleration
- **Impact**: Build times are quick (~30 seconds), and inference is fast enough for web use (~5-10 ms per prediction)

**Complexity vs. Maintainability**:
- **Decision**: Custom neural network architecture vs. simpler models
- **Tradeoff**: More complex model requires careful architecture matching between training and inference
- **Mitigation**: Model metadata (`model_info.pkl`) stores architecture parameters to ensure compatibility

**Development vs. Production**:
- **Decision**: Automatic model training on startup vs. pre-trained models
- **Tradeoff**: Training on startup ensures fresh models but increases startup time
- **Rationale**: Automatic training ensures models are always up-to-date with the latest data

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

### Operations

**Logs/Metrics**:
- Flask development server logs to stdout (captured by Docker)
- Health check endpoint (`/health`) for container monitoring
- **Known Limitation**: No structured logging or metrics collection (suitable for MVP)

**Scaling Considerations**:
- Current design: Single-container deployment
- **Limitation**: No horizontal scaling capability (would require load balancer, shared state)
- **Future**: Could add Redis for session management, gunicorn for multi-worker support

**Resource Footprint**:
- Container image: ~500 MB - 1 GB (Python 3.11 slim base with dependencies)
- Memory usage: ~500 MB - 1 GB at runtime
- CPU: Minimal for inference (single-threaded predictions)
- GPU: Not currently used (CPU-only PyTorch)

**Known Limitations**:
1. Model trains on every container startup (adds ~30-60 seconds to startup time)
2. No database persistence (stateless predictions only)
3. Single-threaded Flask server (not production-grade for high traffic)
4. No authentication/authorization (suitable for local use only)
5. Training data must be provided via volume mount (`DBDData.csv`)

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
  "probability": 0.6745
}
```

**Web Interface:**
- Categorized form sections (Player Characteristics, Items, Map Information, Perks, Bloodpoints)
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
- Container startup: ~30-60 seconds (includes model training)
- Training time: ~15-30 seconds (15 epochs)
- Memory footprint: ~500 MB base + model size
- Disk I/O: Minimal (model loaded once after training)

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
│   ├── app.py             # Flask web application
│   ├── train_model.py     # Model training script
│   └── save_model.py      # Model saving utility
├── tests/                  # Unit and smoke tests
│   ├── test_api.py        # API endpoint tests
│   └── test_model.py      # Model architecture tests
├── templates/              # HTML templates
│   └── index.html         # Main web interface
├── assets/                 # Images and static assets
├── DBDData.csv            # Training data (not in repo)
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── start.sh              # Container startup script
```

### Planned Improvements

   - Collect more data to improve model accuracy
   - Update data as game patches and updates are released
   - Add more sophisticated feature engineering
   - Implement model versioning

### Recent Updates

   - Added comprehensive perk support (exhaustion, chase, other perks)
   - Added map type and powerful add-ons features
   - Organized UI into categorized sections
   - Enhanced interface with icons for better UX
   - Simplified outcome display
   - Reduced training epochs for faster startup
   - Added unit tests
   - Improved code organization (src/ folder structure)

### Stretch Features

   - It would be nice to be able to eventually collaborate with a group like NightLight to actively analyze auto-screenshots from games from the community rather than just my killer games to increase accuracy


## 7) Links (Required)

**GitHub Repo**: https://github.com/nhorton06/DBDPrediction

**Credits & Attribution**: See [CREDITS.md](CREDITS.md) for attribution of game assets and third-party resources.