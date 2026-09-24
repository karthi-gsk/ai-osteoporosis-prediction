import sys
import os
import requests
import json

def run_smoke_test():
    base_url = "http://127.0.0.1:8000"
    print("=== TASK 4: PRODUCTION SMOKE TEST ===", flush=True)
    
    # 1. Test Health Endpoint
    print("\n1. Testing Backend Health Endpoint (/api/health)...", flush=True)
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"Status Code: {r.status_code}", flush=True)
        health_data = r.json()
        print(f"Response: {json.dumps(health_data, indent=2)}", flush=True)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert health_data.get("status") in ["operational", "healthy", "degraded"], f"Unexpected health status: {health_data.get('status')}"
        assert health_data.get("model_loaded") is True, "Model should be loaded"
        assert "ResNet18" in health_data.get("model_identifier", ""), "Expected ResNet18 identifier"
    except Exception as e:
        print(f"FAILED Health Check: {e}", flush=True)
        return False
        
    # 2. Test Real ResNet18 Inference + Genuine Grad-CAM
    val_img = os.path.join("dataset", "strict_clean_knee_osteoporosis", "val", "Normal", "N10.JPEG")
    if not os.path.exists(val_img):
        val_img = os.path.join("dataset", "strict_clean_knee_osteoporosis", "val", "Osteoporosis", "O10.JPEG")
    print(f"\n2. Testing Real Inference & Grad-CAM with: {val_img}...", flush=True)
    
    try:
        with open(val_img, "rb") as f:
            files = {"file": (os.path.basename(val_img), f, "image/jpeg")}
            r = requests.post(f"{base_url}/api/predict", files=files, timeout=15)
            
        print(f"Inference Status Code: {r.status_code}", flush=True)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        pred_data = r.json()
        
        print(f"Predicted Class: {pred_data.get('prediction')}", flush=True)
        print(f"Confidence: {pred_data.get('confidence'):.4f}", flush=True)
        print(f"Class Probabilities: {pred_data.get('class_probabilities')}", flush=True)
        print(f"Measured Latency: {pred_data.get('inference_latency_ms'):.2f} ms", flush=True)
        explainability = pred_data.get("explainability", {})
        print(f"Grad-CAM Status: {explainability.get('status')}", flush=True)
        print(f"Grad-CAM Target Layer: {explainability.get('target_layer')}", flush=True)
        print(f"Grad-CAM Target Class: {explainability.get('target_class')}", flush=True)
        print(f"Grad-CAM Heatmap Present: {explainability.get('heatmap_image') is not None}", flush=True)
        print(f"Grad-CAM Overlay Present: {explainability.get('overlay_image') is not None}", flush=True)
        print(f"Model ID: {pred_data.get('model_identifier')}", flush=True)
        
        # Verify genuine Grad-CAM payload
        assert explainability.get("status") == "success"
        assert explainability.get("heatmap_image") is not None
        assert explainability.get("overlay_image") is not None
        assert "data:image/png;base64," in explainability.get("overlay_image")
        print("Real ResNet18 inference & Grad-CAM payload verified!", flush=True)
    except Exception as e:
        print(f"FAILED Inference Test: {e}", flush=True)
        return False

    # 3. Test Error Handling (Invalid file type)
    print("\n3. Testing Error Handling on Non-Image Upload...", flush=True)
    try:
        bad_files = {"file": ("malicious.txt", b"This is not an image file.", "text/plain")}
        r = requests.post(f"{base_url}/api/predict", files=bad_files, timeout=5)
        print(f"Bad Request Status Code: {r.status_code}", flush=True)
        print(f"Bad Request Error Response: {r.text}", flush=True)
        assert r.status_code in [400, 415, 422], f"Expected client error 400/415/422, got {r.status_code}"
        print("Error handling verified: API correctly rejects non-image files with 4xx!", flush=True)
    except Exception as e:
        print(f"FAILED Error Handling Test: {e}", flush=True)
        return False
        
    print("\nAll Backend Smoke Tests Passed!", flush=True)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
