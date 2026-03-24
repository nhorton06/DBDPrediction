# syntax=docker/dockerfile:1
# Builder: install deps and train. Runtime: bookworm slim only — no gcc/binutils in final image.
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
# pip>=26: CVE-2026-1703; CPU torch index shrinks image vs default CUDA wheels
RUN pip install --no-cache-dir --upgrade "pip>=26.0" setuptools wheel && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple/ \
      -r requirements.txt

COPY src/app.py src/train_model.py src/save_model.py ./
COPY start.sh ./
COPY templates/ ./templates/
COPY assets/ ./assets/
COPY DBDData.csv ./

RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

RUN MODEL_OUTPUT_DIR=/app TRAINING_CSV=/app/DBDData.csv python train_model.py && \
    ls -lh /app/*.pth /app/*.pkl 2>/dev/null || true

# Runtime: same Debian major as builder; toolchain not installed (avoids most binutils CVEs)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    FLASK_HOST=0.0.0.0 \
    FLASK_DEBUG=False

# Full dist upgrade (bookworm-security); upgrade /usr/local pip+wheel so Hub doesn’t flag base 24.0 / wheel 0.45.1 beside the venv
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/python3.11 -m pip install --no-cache-dir --upgrade "pip>=26.0" "wheel>=0.46.2"

COPY --from=builder /venv /venv
COPY --from=builder /app /app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

CMD ["/bin/bash", "./start.sh"]
