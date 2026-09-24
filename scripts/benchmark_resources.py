import time
import os
import sys
import tracemalloc

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

def benchmark():
    tracemalloc.start()
    
    print("=== TASK 3: RESOURCE & LATENCY BENCHMARKING ===", flush=True)
    
    # Measure baseline memory
    current, peak = tracemalloc.get_traced_memory()
    print(f"Initial Python allocated memory: {current / (1024*1024):.2f} MB (Peak: {peak / (1024*1024):.2f} MB)", flush=True)
    
    # Measure cold startup time (import + model load)
    t0 = time.perf_counter()
    from backend.services.ai_service import (
        load_trained_model,
        get_model_status,
        predict_osteoporosis,
        ResNet18InferenceManager,
    )
    t_import = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    success = load_trained_model()
    t_load = time.perf_counter() - t1
    
    status = get_model_status()
    current_idle, peak_idle = tracemalloc.get_traced_memory()
    
    print(f"Module import latency: {t_import*1000:.2f} ms", flush=True)
    print(f"Model initialization & weight loading latency: {t_load*1000:.2f} ms", flush=True)
    print(f"Total cold startup time: {(t_import + t_load)*1000:.2f} ms", flush=True)
    print(f"Model loaded status: {status['is_loaded']}", flush=True)
    print(f"Model metadata: {status['metadata']}", flush=True)
    print(f"Idle memory allocated: {current_idle / (1024*1024):.2f} MB (Peak: {peak_idle / (1024*1024):.2f} MB)", flush=True)
    
    # Validation image for testing
    val_img_path = os.path.join("dataset", "strict_clean_knee_osteoporosis", "val", "Normal", "N10.JPEG")
    if not os.path.exists(val_img_path):
        val_img_path = os.path.join("dataset", "strict_clean_knee_osteoporosis", "val", "Osteoporosis", "O10.JPEG")
        
    print(f"\nUsing test validation image: {val_img_path}", flush=True)
    
    with open(val_img_path, "rb") as f:
        img_bytes = f.read()
        
    meta = {
        "width": 224,
        "height": 224,
        "channels": 3,
        "format": "JPEG",
        "file_size_bytes": len(img_bytes)
    }
    
    # Warmup inference + Grad-CAM run
    _ = predict_osteoporosis(image_bytes=img_bytes, filename=os.path.basename(val_img_path), metadata=meta)
    
    latencies = []
    peak_mems = []
    
    # 5 runs to measure steady-state CPU latency and peak memory during inference + Grad-CAM
    for i in range(5):
        tracemalloc.reset_peak()
        t_start = time.perf_counter()
        resp = predict_osteoporosis(image_bytes=img_bytes, filename=os.path.basename(val_img_path), metadata=meta)
        t_end = time.perf_counter()
        
        curr_m, peak_m = tracemalloc.get_traced_memory()
        lat_ms = (t_end - t_start) * 1000
        latencies.append(lat_ms)
        peak_mems.append(peak_m / (1024 * 1024))
        
        print(f"Run {i+1}: Latency = {lat_ms:.2f} ms, Peak Traced Mem = {peak_m / (1024*1024):.2f} MB, Pred = {resp.prediction} ({resp.confidence*100:.1f}%)", flush=True)
        
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    overall_peak = max(peak_mems)
    
    print(f"\n--- Resource Summary ---", flush=True)
    print(f"Idle Allocated Memory: {current_idle / (1024*1024):.2f} MB", flush=True)
    print(f"Peak Inference + Grad-CAM Memory: {overall_peak:.2f} MB", flush=True)
    print(f"Memory overhead during inference: {overall_peak - (current_idle / (1024*1024)):.2f} MB", flush=True)
    print(f"Average CPU Inference + Grad-CAM Latency: {avg_latency:.2f} ms", flush=True)
    print(f"Min Latency: {min_latency:.2f} ms | Max Latency: {max_latency:.2f} ms", flush=True)
    
    print("\n=== TASK 2: TRUTHFUL ERROR STATE VERIFICATION ===", flush=True)
    # Test what happens when checkpoint path does not exist
    test_manager = ResNet18InferenceManager()
    res = test_manager.load_model(checkpoint_path="non_existent_model_checkpoint.pth")
    print(f"Non-existent checkpoint load result: {res} (Expected False)", flush=True)
    print(f"Truthful error message recorded: {test_manager.error_message}", flush=True)
    assert res is False, "Expected load_model to return False for missing checkpoint"
    assert "not found" in test_manager.error_message.lower(), "Expected error message to mention 'not found'"
    print("Truthful error state verified successfully!", flush=True)

if __name__ == "__main__":
    benchmark()
