from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, Response, status
from backend.models.schemas import PredictionResponse, HealthResponse
from backend.utils.image_validator import validate_image_file
from backend.services.ai_service import predict_osteoporosis, get_model_status

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.get("/health", response_model=HealthResponse, summary="Backend Health Check")
async def health_check(response: Response):
    """Returns operational status and loaded model telemetry. Fails with HTTP 503 if model is not loaded."""
    model_state = get_model_status()
    is_loaded = model_state.get("is_loaded", False)
    meta = model_state.get("metadata", {})

    if not is_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    best_epoch = meta.get("best_epoch")
    best_f1 = meta.get("best_val_macro_f1")
    model_id = f"ResNet18 (best_model.pth, Epoch {best_epoch})" if is_loaded else None

    return HealthResponse(
        status="operational" if is_loaded else "degraded",
        version="1.0.0-phase5b",
        service="AI Osteoporosis Prediction Backend",
        model_loaded=is_loaded,
        model_identifier=model_id,
        checkpoint_best_epoch=best_epoch if isinstance(best_epoch, int) else None,
        checkpoint_best_val_f1=round(best_f1, 4) if isinstance(best_f1, float) else None,
        device=model_state.get("device", "cpu")
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Knee X-ray for Osteoporosis Risk"
)
async def predict_bone_health(
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(
        None,
        ge=0,
        le=2,
        description="Optional target class for Grad-CAM (0=Normal, 1=Osteopenia, 2=Osteoporosis). Defaults to predicted class."
    )
):
    """
    Accepts an uploaded knee X-ray image (JPG, JPEG, PNG),
    validates file integrity, size, and dimensions,
    and returns genuine ResNet18 prediction with Grad-CAM explainability.
    """
    # Read image contents into bytes
    file_bytes = await file.read()

    # Validate image format, mime-type, dimensions, and integrity
    metadata = validate_image_file(file, file_bytes)

    # Invoke genuine ResNet18 prediction service with Grad-CAM
    prediction_result = predict_osteoporosis(
        image_bytes=file_bytes,
        filename=file.filename or "uploaded_xray.png",
        metadata=metadata,
        target_class=target_class
    )

    return prediction_result
