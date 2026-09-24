from pydantic import BaseModel, Field
from typing import Dict, Optional


class ClassProbabilities(BaseModel):
    Normal: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for Normal category")
    Osteopenia: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for Osteopenia category")
    Osteoporosis: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for Osteoporosis category")


class ExplainabilityMetadata(BaseModel):
    method: str = "Grad-CAM (Gradient-weighted Class Activation Mapping)"
    status: str = Field("success", description="Status of explainability generation: success, error, or disabled")
    message: str = Field(..., description="Details regarding the explainability analysis")
    target_layer: Optional[str] = Field(None, description="Hooked CNN layer name and spatial resolution")
    target_class: Optional[str] = Field(None, description="Class name for which Grad-CAM gradients were computed")
    target_class_index: Optional[int] = Field(None, description="Index of target class (0=Normal, 1=Osteopenia, 2=Osteoporosis)")
    overlay_image: Optional[str] = Field(None, description="Base64 Data URI string of the transparent Grad-CAM overlay")
    heatmap_image: Optional[str] = Field(None, description="Base64 Data URI string of the standalone heatmap")


class FileInfo(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    dimensions: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted bone category: Normal, Osteopenia, or Osteoporosis")
    predicted_class_index: int = Field(..., description="Class index: 0=Normal, 1=Osteopenia, 2=Osteoporosis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax score for the top predicted category")
    risk_level: str = Field(..., description="Clinical risk category: Low (Normal), Moderate (Osteopenia), High (Osteoporosis)")
    model_status: str = Field(..., description="Status of the deep learning model")
    model_identifier: str = Field("ResNet18", description="Architecture and checkpoint details")
    inference_status: str = Field("success", description="Inference pipeline status")
    inference_latency_ms: float = Field(..., description="Actual measured inference latency in milliseconds")
    analysis_date: str = Field(..., description="ISO formatted timestamp of analysis")
    class_probabilities: ClassProbabilities
    explainability: ExplainabilityMetadata
    file_info: FileInfo
    disclaimer: str = (
        "Academic & Research Notice: This model was trained and evaluated exclusively on knee X-ray radiographs. "
        "It does NOT evaluate hip, lumbar spine, or femoral neck anatomies. "
        "Softmax scores are mathematical model outputs, not calibrated diagnostic probabilities. "
        "Grad-CAM heatmaps highlight feature attribution, not medically verified bone mineral loss. "
        "This tool does not measure BMD or DXA T-scores and must not be used as a medical diagnostic system."
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    model_loaded: bool
    model_identifier: Optional[str] = None
    checkpoint_best_epoch: Optional[int] = None
    checkpoint_best_val_f1: Optional[float] = None
    device: Optional[str] = None
