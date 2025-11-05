# Deployment Guide

This guide covers deploying the DBD Escape Prediction API to various platforms.

## 📦 GitHub Setup

### 1. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: DBD Escape Prediction API"
```

### 2. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. **Do NOT** initialize with README (we already have one)
3. Copy the repository URL

### 3. Push to GitHub

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

### 4. Important Notes for GitHub

- **Model files are excluded**: The `.gitignore` file excludes `dbd_model.pth`, `scaler.pkl`, and `model_info.pkl`
- Users must train their own model or you can provide instructions
- To share trained models (if desired), use GitHub Releases or a separate storage solution

## 🐳 Docker Deployment

### Local Docker Testing

```bash
# Build the image
docker build -t dbd-predictor .

# Run with model files mounted
docker run -d -p 5000:5000 \
  -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
  -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
  -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
  --name dbd-predictor dbd-predictor

# Or use docker-compose
docker-compose up -d
```

### Push to Docker Hub

1. **Create Docker Hub account**: https://hub.docker.com

2. **Login to Docker Hub**
   ```bash
   docker login
   ```

3. **Tag your image**
   ```bash
   docker tag dbd-predictor yourusername/dbd-predictor:latest
   ```

4. **Push to Docker Hub**
   ```bash
   docker push yourusername/dbd-predictor:latest
   ```

5. **Others can now pull and run**
   ```bash
   docker pull yourusername/dbd-predictor:latest
   docker run -d -p 5000:5000 \
     -v $(pwd)/dbd_model.pth:/app/dbd_model.pth:ro \
     -v $(pwd)/scaler.pkl:/app/scaler.pkl:ro \
     -v $(pwd)/model_info.pkl:/app/model_info.pkl:ro \
     yourusername/dbd-predictor:latest
   ```

## ☁️ Cloud Platform Deployment

### Railway.app

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Add model files as secrets or upload them
5. Deploy: `railway up`

### Render.com

1. Connect your GitHub repository
2. Create a new Web Service
3. Set build command: `docker build -t dbd-predictor .`
4. Set start command: `docker run -p $PORT:5000 dbd-predictor`
5. Add model files via environment variables or file upload

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/dbd-predictor

# Deploy
gcloud run deploy dbd-predictor \
  --image gcr.io/PROJECT-ID/dbd-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS Elastic Container Service (ECS)

1. Build and push to Amazon ECR
2. Create ECS cluster and service
3. Configure task definition with model file volumes
4. Deploy and access via ALB

### Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry <registry-name> --image dbd-predictor .

# Deploy
az container create \
  --resource-group <resource-group> \
  --name dbd-predictor \
  --image <registry-name>.azurecr.io/dbd-predictor \
  --dns-name-label dbd-predictor \
  --ports 5000
```

## 🔐 Managing Model Files in Production

### Option 1: Volume Mounts (Local/Docker)
- Mount model files as read-only volumes
- Best for local deployment or when you have persistent storage

### Option 2: Environment Variables (Small files)
- Convert to base64 and store as environment variables
- Not recommended for large model files

### Option 3: Cloud Storage
- Upload models to S3, GCS, or Azure Blob Storage
- Download at container startup
- Best for production deployments

### Option 4: Container Image
- Include models in the Docker image (increases image size)
- Update the Dockerfile to COPY model files
- Simplest but less flexible

Example for Option 4 (modify Dockerfile):
```dockerfile
# Add before CMD
COPY dbd_model.pth scaler.pkl model_info.pkl /app/
```

## 📊 Monitoring and Logging

### Add Health Check Endpoint

Add to `app.py`:

```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and scaler is not None
    }), 200
```

### Docker Health Check

Already included in Dockerfile, but you can customize:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:5000/health || exit 1
```

## 🚀 CI/CD with GitHub Actions

The included `.github/workflows/docker-build.yml` will:
- Build Docker image on push/PR
- Test that image builds successfully
- Can be extended to push to registries

## 📝 Environment Variables

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Host to bind to |
| `FLASK_PORT` | `5000` | Port to listen on |
| `FLASK_DEBUG` | `False` | Enable debug mode |

## 🐛 Troubleshooting

### Container won't start
- Check model files are present and mounted correctly
- Review container logs: `docker logs dbd-predictor`
- Verify file permissions

### Port conflicts
- Change host port: `-p 8080:5000`
- Or change container port via `FLASK_PORT`

### Model loading errors
- Ensure all three model files are present
- Check file paths in volume mounts
- Verify file permissions are readable

### Out of memory
- PyTorch can be memory-intensive
- Increase container memory limits
- Consider using a smaller model or batch size

