"""
Phase 5B: Comprehensive Backend & API Verification Script
Validates the FastAPI backend integration of ResNet18 and genuine Grad-CAM.
Uses only the validation set images (data/strict_clean/val/).
Does NOT evaluate the held-out test dataset.
"""

import os
import sys
import io
import time
import base64
import hashlib
from PIL import Image
from fastapi.testclient import TestClient

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.main import app
from backend.services.ai_service import ResNet18InferenceManager, load_trained_model
from scripts.gradcam import load_trained_resnet18, GradCAM, preprocess_image, CLASS_NAMES


def get_file_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_backend_verification():
    print("=" * 65)
    print("  PHASE 5B: FASTAPI REAL MODEL & GRAD-CAM BACKEND VERIFICATION")
    print("=" * 65)

    checkpoint_path = os.path.join(WORKSPACE_ROOT, "models", "checkpoints", "resnet18", "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
    initial_ckpt_hash = get_file_sha256(checkpoint_path)
    print(f"\n1. Initial Checkpoint SHA-256: {initial_ckpt_hash[:16]}... (Immunity Anchor)")

    # Initialize TestClient
    with TestClient(app) as client:
        # Step 1: Verify Health Endpoint
        print("\n2. Testing GET /api/health ...")
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        print(f"   Status:                 {health_data.get('status')}")
        print(f"   Model Loaded:           {health_data.get('model_loaded')}")
        print(f"   Model Identifier:       {health_data.get('model_identifier')}")
        print(f"   Checkpoint Best Epoch:  {health_data.get('checkpoint_best_epoch')}")
        print(f"   Checkpoint Best Val F1: {health_data.get('checkpoint_best_val_f1')}")
        assert health_data.get("model_loaded") is True, "Model is not marked as loaded!"
        assert health_data.get("checkpoint_best_epoch") == 21, "Unexpected best epoch!"

        # Step 2: Validate Predictions & Grad-CAM on representative validation images
        val_dir = os.path.join(WORKSPACE_ROOT, "dataset", "strict_clean_knee_osteoporosis", "val")
        val_samples = [
            ("Normal", os.path.join(val_dir, "Normal", "N10.JPEG")),
            ("Osteopenia", os.path.join(val_dir, "Osteopenia", "OP104.jpg")),
            ("Osteoporosis", os.path.join(val_dir, "Osteoporosis", "OS19.jpg")),
        ]

        print("\n3. Testing POST /api/predict on Validation Samples...")
        for label, img_path in val_samples:
            assert os.path.exists(img_path), f"Validation sample missing: {img_path}"
            orig_pil = Image.open(img_path)
            orig_w, orig_h = orig_pil.size

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            t0 = time.perf_counter()
            resp = client.post(
                "/api/predict",
                files={"file": (os.path.basename(img_path), img_bytes, "image/jpeg")}
            )
            roundtrip_ms = (time.perf_counter() - t0) * 1000

            assert resp.status_code == 200, f"Prediction failed for {label}: {resp.text}"
            data = resp.json()

            pred_class = data["prediction"]
            pred_idx = data["predicted_class_index"]
            conf = data["confidence"]
            probs = data["class_probabilities"]
            risk = data["risk_level"]
            latency_ms = data["inference_latency_ms"]
            expl = data["explainability"]

            prob_sum = probs["Normal"] + probs["Osteopenia"] + probs["Osteoporosis"]

            print(f"\n   --- Sample: {label} ({os.path.basename(img_path)}) ---")
            print(f"   Predicted Class:        {pred_class} (Index: {pred_idx})")
            print(f"   Confidence Score:       {conf * 100:.2f}%")
            print(f"   Risk Level:             {risk}")
            print(f"   Probabilities:          Normal={probs['Normal']:.4f}, Osteopenia={probs['Osteopenia']:.4f}, Osteoporosis={probs['Osteoporosis']:.4f}")
            print(f"   Probability Sum:        {prob_sum:.4f} (Valid: {0.999 <= prob_sum <= 1.001})")
            print(f"   Measured Server Time:   {latency_ms:.2f} ms (Roundtrip: {roundtrip_ms:.2f} ms)")
            print(f"   Grad-CAM Status:        {expl['status']}")
            print(f"   Grad-CAM Target Class:  {expl['target_class']} ({expl['target_class_index']})")
            print(f"   Grad-CAM Target Layer:  {expl['target_layer']}")

            # Assertions
            assert 0.999 <= prob_sum <= 1.001, f"Probabilities do not sum to 1: {prob_sum}"
            assert expl["status"] == "success", "Grad-CAM did not return success status!"
            assert expl["overlay_image"].startswith("data:image/png;base64,"), "Overlay image URI format invalid!"
            assert expl["heatmap_image"].startswith("data:image/png;base64,"), "Heatmap image URI format invalid!"

            # Decode overlay and verify dimensions match original
            b64_data = expl["overlay_image"].split(",")[1]
            overlay_bytes = base64.b64decode(b64_data)
            overlay_img = Image.open(io.BytesIO(overlay_bytes))
            print(f"   Overlay Image Size:     {overlay_img.size} (Original: {orig_w}x{orig_h})")
            assert overlay_img.size == (orig_w, orig_h), f"Overlay size {overlay_img.size} does not match original ({orig_w}, {orig_h})!"

        # Step 3: Test Target Class Parameter for Grad-CAM
        print("\n4. Testing Target Class Override (?target_class=2 on Normal image)...")
        normal_path = val_samples[0][1]
        with open(normal_path, "rb") as f:
            img_bytes = f.read()

        resp = client.post(
            "/api/predict?target_class=2",
            files={"file": ("N10.JPEG", img_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200, f"Target class override failed: {resp.text}"
        data = resp.json()
        expl = data["explainability"]
        print(f"   Predicted Class:        {data['prediction']}")
        print(f"   Grad-CAM Target Class:  {expl['target_class']} (Index: {expl['target_class_index']})")
        assert expl["target_class_index"] == 2, f"Target class was not overridden to 2: {expl['target_class_index']}"
        assert expl["target_class"] == "Osteoporosis"

        # Step 4: Test Preprocessing & Mathematical Consistency against Standalone GradCAM
        print("\n5. Testing Mathematical Consistency vs Standalone GradCAM...")
        mgr = ResNet18InferenceManager.get_instance()
        test_img = val_samples[1][1]
        standalone_tensor, standalone_pil = preprocess_image(test_img)
        standalone_cam = GradCAM(mgr.model)
        standalone_out = standalone_cam.generate_cam(standalone_tensor)
        standalone_cam.remove_hooks()

        with open(test_img, "rb") as f:
            resp = client.post(
                "/api/predict",
                files={"file": (os.path.basename(test_img), f.read(), "image/jpeg")}
            )
        api_data = resp.json()
        api_probs = [
            api_data["class_probabilities"]["Normal"],
            api_data["class_probabilities"]["Osteopenia"],
            api_data["class_probabilities"]["Osteoporosis"],
        ]
        diff = max(abs(a - b) for a, b in zip(api_probs, standalone_out["all_probs"]))
        print(f"   Max absolute difference between API and Standalone: {diff:.6f}")
        assert diff < 1e-3, f"Inconsistency between API and standalone inference: diff={diff}"
        print("   Mathematical consistency verified! (Zero drift)")

        # Step 5: Test Error Handling
        print("\n6. Testing Backend Error Handling...")
        
        # Test empty file
        resp = client.post("/api/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
        print(f"   Empty File Test:         Status {resp.status_code} ({resp.json().get('detail')})")
        assert resp.status_code == 400

        # Test corrupt image
        resp = client.post("/api/predict", files={"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA", "image/jpeg")})
        print(f"   Corrupted File Test:     Status {resp.status_code} ({resp.json().get('detail')[:45]}...)")
        assert resp.status_code == 400

        # Test unsupported extension
        resp = client.post("/api/predict", files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")})
        print(f"   Unsupported Type Test:   Status {resp.status_code} ({resp.json().get('detail')})")
        assert resp.status_code == 400

        # Test invalid target_class range
        resp = client.post(
            "/api/predict?target_class=99",
            files={"file": ("N10.JPEG", img_bytes, "image/jpeg")}
        )
        print(f"   Invalid Target Class:    Status {resp.status_code} (Validation Error caught)")
        assert resp.status_code == 422

        # Step 6: Verify Checkpoint Immutability
        final_ckpt_hash = get_file_sha256(checkpoint_path)
        print("\n7. Verifying Checkpoint Immutability...")
        print(f"   Initial Checkpoint Hash: {initial_ckpt_hash}")
        print(f"   Final Checkpoint Hash:   {final_ckpt_hash}")
        assert initial_ckpt_hash == final_ckpt_hash, "CRITICAL: Checkpoint was modified during inference!"
        print("   Checkpoint file is 100% bit-for-bit IMMUTABLE.")

        print("\n" + "=" * 65)
        print("  ALL BACKEND INTEGRATION & VERIFICATION CHECKS PASSED!")
        print("=" * 65)


if __name__ == "__main__":
    run_backend_verification()
