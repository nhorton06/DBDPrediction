# Dead by Daylight Escape Prediction - Case Study

## 1) Executive Summary

### Problem

Dead by Daylight (DBD) players, especially ones like myself who have dedicated thousands of hours to the game, want to understand a survivor's chance of escaping a match based on various in-game factors. Currently, I basically just have to rely on intuition and my personal experience to estimate escape probability. I can also sometimes look at a survivor's hour count in-game, but this is not always available as it is entirely up to the player to make this public or not. This project addresses the need for a data-driven prediction tool that helps potentially make more informed decisions during pre-game lobby queue and understand how certain factors (survivor characteristics, map size, items, prestige level) may or may not influence escape rates.

**Target Users**: Dead by Daylight players, gaming analysts, and researchers interested in game mechanics analysis

### Solution

This project is a machine learning web application that predicts a survivor's probability of escaping in Dead by Daylight using a PyTorch neural network. The system accepts inputs such as survivor gender, prestige level, items brought, map characteristics, and player statistics (bloodpoints), then outputs a probability score between 0-100% indicating the likelihood/confidence of the model's outcome of escape or death. The application is delivered as a Flask web app with a user-friendly interface deployed through a container on Docker, making it accessible to both technical and non-technical users. The model was trained on my personal killer gameplay data to learn patterns that correlate with successful escapes.

## 2) System Overview

### Course Concept(s)

- **Flask API**: Putting my neural network model into a Flask app for deployment into an html file to be interactive for the user
- **Containerization**: Docker for consistent deployment and environment management

### Architecture Diagram

![System Architecture](assets/architecture.png)

*Note: Place architecture diagram PNG in `/assets/architecture.png`*

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
- **Input Features**: 13 features (gender, steam player, anonymous mode, prestige, map area, item type, survivor/killer bloodpoints)
- **Output**: Binary classification probability (escape vs. no escape)
- **Format**: Saved as `dbd_model.pth` (PyTorch state dict), `scaler.pkl` (scikit-learn StandardScaler), `model_info.pkl` (architecture metadata)
- **Size**: Model weights ~few MB, total artifacts ~10-50 MB

**Services:**
- **Web Interface**: Flask web server (port 5000)
- **API Endpoint**: POST `/predict` for programmatic access
- **Health Check**: GET `/health` for container health monitoring

## 3) How to Run (Local)

### Docker

```bash
# build
docker build -t dbd-predictor:latest .

# run
docker run --rm -p 5000:5000 \
  -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
  -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
  -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
  dbd-predictor:latest

# health check
curl http://localhost:5000/health
```

**Alternative: Using Docker Compose**

```bash
# build and run
docker-compose up -d

# health check
curl http://localhost:5000/health

# view logs
docker-compose logs -f

# stop
docker-compose down
```

## 4) Design Decisions

### Why This Concept?

**Docker**: Essential for ensuring consistent deployment across different environments. The container includes all dependencies (PyTorch, Flask, scikit-learn) and eliminates "works on my machine" issues. CUDA-enabled PyTorch is included to support GPU acceleration when available, while gracefully falling back to CPU.

**Flask**: Chosen for its simplicity and lightweight nature. The application only needs basic routing and JSON handling, making Flask more appropriate than heavier frameworks like Django. The REST API design allows for both web interface and programmatic access.

### Tradeoffs

**Performance vs. Model Size**: 
- **Decision**: Full CUDA-enabled PyTorch (~4.5 GB) vs. CPU-only (~200 MB)
- **Tradeoff**: Chose CUDA version for GPU acceleration capability, accepting longer build times and larger image size
- **Impact**: First build takes 10-20 minutes, but subsequent builds are cached. GPU inference is 10-100x faster when available.

**Complexity vs. Maintainability**:
- **Decision**: Custom neural network architecture vs. simpler models
- **Tradeoff**: More complex model requires careful architecture matching between training and inference
- **Mitigation**: Model metadata (`model_info.pkl`) stores architecture parameters to ensure compatibility

**Development vs. Production**:
- **Decision**: Model files as volumes vs. baked into image
- **Tradeoff**: Volumes allow model updates without rebuilding, but require file management
- **Rationale**: Models are large and change infrequently, so volume mounting is appropriate

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
- Container image: ~5 GB (with CUDA PyTorch)
- Memory usage: ~500 MB - 1 GB at runtime
- CPU: Minimal for inference (single-threaded predictions)
- GPU: Optional, provides significant speedup if available

**Known Limitations**:
1. Model files must be provided externally (not in image)
2. No database persistence (stateless predictions only)
3. Single-threaded Flask server (not production-grade for high traffic)
4. No authentication/authorization (suitable for local use only)

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
- Displays escape probability as percentage
- Color-coded results (green for escape, red for death)
- Visual feedback with animations and glow effects

### Performance Notes

**Inference Speed**:
- CPU: ~5-10 ms per prediction
- GPU (if available): ~1-2 ms per prediction
- Network latency: Minimal (local deployment)

**Resource Usage**:
- Container startup: ~10-15 seconds (model loading)
- Memory footprint: ~500 MB base + model size
- Disk I/O: Minimal (model loaded once at startup)

### Validation/Tests

**Model Validation**:
- Trained model evaluated on test set (typically 20% holdout)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Cross-validation performed during training (see `DBDCode.ipynb`)

**API Testing**:
- Health check endpoint verified
- Input validation tested with edge cases (invalid ranges, missing fields)
- Error handling verified for malformed requests

**Integration Testing**:
- Docker container build verified
- Volume mounting tested for model file access
- End-to-end workflow: form submission → prediction → display

## 6) What's Next

### Planned Improvements

   - Incorporate additional features (map type, powerful perks, add-ons)
   - Collect more data to train and test off for better accuracy
   - Update data as more updates and patches come out to the game itself

### Refactors

   - Better code organization
   - Unit tests for prediction logic

### Stretch Features

   - It would be nice to be able to eventually collaborate with a group like NightLight to actively analyze auto-screenshots from games from the community rather than just my killer games to increase accuracy
   - 


## 7) Links (Required)

**GitHub Repo**: https://github.com/nhorton06/DBDPrediction