import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.services.ai_service import load_trained_model

app = FastAPI(
    title="AI Osteoporosis Prediction System API",
    description=(
        "Backend API for knee X-ray analysis and osteoporosis risk prediction using "
        "a trained ResNet18 model and genuine Grad-CAM explainability. Academic & Research prototype."
    ),
    version="1.0.0"
)

# Configure CORS via environment variable with local development defaults
raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,*"
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Triggered when FastAPI starts up - loads ResNet18 weights into memory once."""
    load_trained_model()


@app.get("/")
async def root():
    return {
        "name": "AI Osteoporosis Prediction System API",
        "version": "1.0.0",
        "status": "online",
        "target_anatomy": "Knee Radiographs (Bilateral / Unilateral AP)",
        "model": "ResNet18 (Frozen Checkpoint)",
        "explainability": "Genuine Grad-CAM (layer4[-1])",
        "docs_url": "/docs",
        "predict_endpoint": "/api/predict",
        "health_endpoint": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENV", "development").lower() == "development"
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)
