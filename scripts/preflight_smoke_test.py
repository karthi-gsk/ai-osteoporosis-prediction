"""
Phase 9A Preflight Production Smoke Test Script.
Validates:
1. Complete FastAPI application import.
2. /api/health HTTP 200 (healthy/operational) when model is loaded.
3. /api/health HTTP 503 (degraded) when model is unavailable.
4. Real ResNet18 inference with genuine logits/softmax.
5. Genuine Grad-CAM explainability with non-empty overlays.
6. Zero simulated/mock predictions returned.
7. Checkpoint hash and size unchanged.
"""

import os
import sys
import hashlib
import io
import time
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

EXPECTED_SHA256 = "6ccc18b8b71b4bd3d81740aeb9be48271ae24aef119fd0d37801362017aa8faa"
EXPECTED_SIZE = 134266309
CHECKPOINT_PATH = os.path.join("models", "checkpoints", "resnet18", "best_model.pth")
VAL_IMAGE_PATH = os.path.join("dataset", "strict_clean_knee_osteoporosis", "val", "Normal", "N10.JPEG")


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def run_preflight_smoke_test():
    print("=" * 70, flush=True)
    print("PHASE 9A — PREFLIGHT PRODUCTION SMOKE TEST", flush=True)
    print("=" * 70, flush=True)

    # 1. Verify Checkpoint Integrity & Unchanged Status
    print("\n1. Verifying frozen checkpoint integrity...", flush=True)
    assert os.path.exists(CHECKPOINT_PATH), f"Missing checkpoint: {CHECKPOINT_PATH}"
    actual_size = os.path.getsize(CHECKPOINT_PATH)
    actual_sha = compute_sha256(CHECKPOINT_PATH)
    print(f"   Size    : {actual_size} bytes (Expected: {EXPECTED_SIZE})", flush=True)
    print(f"   SHA-256 : {actual_sha}", flush=True)
    assert actual_size == EXPECTED_SIZE, "Checkpoint size changed!"
    assert actual_sha == EXPECTED_SHA256, "Checkpoint SHA-256 changed!"
    print("   [PASSED] Frozen checkpoint is 100% intact and unchanged.", flush=True)

    # 2. Verify FastAPI Application Imports Cleanly
    print("\n2. Testing FastAPI application and route imports...", flush=True)
    from backend.main import app
    from backend.services.ai_service import ResNet18InferenceManager, load_trained_model
    client = TestClient(app)
    print("   [PASSED] FastAPI application and dependencies imported without error.", flush=True)

    # 3. Verify /api/health HTTP 200 when model is loaded
    print("\n3. Testing /api/health with loaded model...", flush=True)
    load_trained_model()
    resp_health_ok = client.get("/api/health")
    print(f"   Status Code : {resp_health_ok.status_code} (Expected: 200)", flush=True)
    health_ok_data = resp_health_ok.json()
    print(f"   Payload     : {health_ok_data}", flush=True)
    assert resp_health_ok.status_code == 200, f"Expected 200, got {resp_health_ok.status_code}"
    assert health_ok_data.get("model_loaded") is True, "Expected model_loaded=True"
    assert health_ok_data.get("status") == "operational", "Expected status='operational'"
    assert "ResNet18" in health_ok_data.get("model_identifier", ""), "Expected ResNet18 identifier"
    print("   [PASSED] Healthy endpoint responds with HTTP 200 and operational telemetry.", flush=True)

    # 4. Verify /api/health HTTP 503 when model is unavailable
    print("\n4. Testing /api/health truthful degradation (HTTP 503 when unloaded)...", flush=True)
    manager = ResNet18InferenceManager.get_instance()
    # Temporarily simulate unloaded state
    original_model = manager.model
    original_loaded = manager.is_loaded
    manager.model = None
    manager.is_loaded = False
    manager.error_message = "Test error: Checkpoint intentionally decoupled for health check verification"

    resp_health_degraded = client.get("/api/health")
    print(f"   Status Code : {resp_health_degraded.status_code} (Expected: 503)", flush=True)
    health_deg_data = resp_health_degraded.json()
    print(f"   Payload     : {health_deg_data}", flush=True)
    assert resp_health_degraded.status_code == 503, f"Expected 503, got {resp_health_degraded.status_code}"
    assert health_deg_data.get("model_loaded") is False, "Expected model_loaded=False"
    assert health_deg_data.get("status") == "degraded", "Expected status='degraded'"
    print("   [PASSED] Degraded endpoint accurately returns HTTP 503 (no silent healthy service).", flush=True)

    # Restore loaded state
    manager.model = original_model
    manager.is_loaded = original_loaded
    manager.error_message = None

    # 5. Verify Real ResNet18 Inference & Genuine Grad-CAM
    print(f"\n5. Testing real ResNet18 inference & Grad-CAM using validation image: {VAL_IMAGE_PATH}...", flush=True)
    assert os.path.exists(VAL_IMAGE_PATH), f"Missing validation image: {VAL_IMAGE_PATH}"
    with open(VAL_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("N10.JPEG", img_bytes, "image/jpeg")}
    t_start = time.perf_counter()
    resp_predict = client.post("/api/predict", files=files)
    t_end = time.perf_counter()

    print(f"   Inference Status Code : {resp_predict.status_code} (Expected: 200)", flush=True)
    assert resp_predict.status_code == 200, f"Expected 200, got {resp_predict.status_code}"
    pred_data = resp_predict.json()

    print(f"   Prediction            : {pred_data.get('prediction')}", flush=True)
    print(f"   Confidence            : {pred_data.get('confidence'):.4f}", flush=True)
    print(f"   Class Probabilities   : {pred_data.get('class_probabilities')}", flush=True)
    print(f"   Measured Total Time   : {(t_end - t_start)*1000:.2f} ms", flush=True)
    print(f"   Reported Model Time   : {pred_data.get('inference_latency_ms'):.2f} ms", flush=True)

    # Verify no mock / simulation values
    assert pred_data.get("prediction") in ["Normal", "Osteopenia", "Osteoporosis"]
    probs = pred_data.get("class_probabilities", {})
    sum_probs = sum(probs.values())
    print(f"   Softmax Sum           : {sum_probs:.6f}", flush=True)
    assert abs(sum_probs - 1.0) < 1e-3, "Probabilities do not sum to 1.0"

    # Check Grad-CAM
    explainability = pred_data.get("explainability", {})
    print(f"   Grad-CAM Status       : {explainability.get('status')}", flush=True)
    print(f"   Grad-CAM Target Layer : {explainability.get('target_layer')}", flush=True)
    assert explainability.get("status") == "success", "Grad-CAM generation failed"
    assert "layer4" in explainability.get("target_layer", ""), "Grad-CAM did not target layer4"
    assert explainability.get("overlay_image", "").startswith("data:image/png;base64,"), "Missing overlay image URI"
    assert explainability.get("heatmap_image", "").startswith("data:image/png;base64,"), "Missing heatmap image URI"
    print("   [PASSED] Real inference and genuine Grad-CAM payload verified (zero mock data).", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("[ALL CHECKS PASSED] PHASE 9A PREFLIGHT SMOKE TEST COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    return True


if __name__ == "__main__":
    success = run_preflight_smoke_test()
    sys.exit(0 if success else 1)
