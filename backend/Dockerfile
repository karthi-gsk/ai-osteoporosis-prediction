# Production Dockerfile for AI Osteoporosis Prediction System — FastAPI Backend
FROM python:3.10-slim

WORKDIR /app

# Install curl for container health monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch and FastAPI dependencies
COPY backend/requirements-cpu.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete backend modules, Grad-CAM scripts, and model structure
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_CHECKPOINT_PATH=/app/models/checkpoints/resnet18/best_model.pth
ENV ENV=production

EXPOSE 8000

# Dynamic port-aware health check: checks configured $PORT with fallback to 8000.
# Fails with exit 1 if /api/health returns HTTP 503 (model not loaded) or server is unreachable.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/api/health" || exit 1

# Enforce checkpoint verification/download before starting Uvicorn.
# If download/verification fails, server starts in degraded mode where /api/health returns HTTP 503,
# immediately alerting container orchestrators (Render, Cloud Run, ECS) that service is unhealthy.
CMD ["sh", "-c", "python scripts/download_checkpoint.py || echo '[WARNING] Checkpoint initialization failed; /api/health will report HTTP 503 degraded state'; python -m uvicorn backend.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
