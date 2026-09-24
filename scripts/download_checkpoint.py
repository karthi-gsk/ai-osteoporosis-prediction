"""
Automated Model Checkpoint Verification and Download Strategy.
Ensures the frozen ResNet18 checkpoint (134 MB) is verified or downloaded
without committing large binary weights directly to normal Git history.
"""

import os
import sys
import hashlib
import urllib.request
import shutil

EXPECTED_SHA256 = "6ccc18b8b71b4bd3d81740aeb9be48271ae24aef119fd0d37801362017aa8faa"
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_PATH = os.path.join(WORKSPACE_ROOT, "models", "checkpoints", "resnet18", "best_model.pth")


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file in 1MB chunks."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def verify_or_download_checkpoint() -> bool:
    print("=" * 70, flush=True)
    print("AI Osteoporosis Prediction System — Model Checkpoint Verification", flush=True)
    print("=" * 70, flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found local checkpoint at: {CHECKPOINT_PATH}", flush=True)
        print("Computing SHA-256 checksum...", flush=True)
        file_hash = compute_sha256(CHECKPOINT_PATH)
        print(f"Computed SHA-256 : {file_hash}", flush=True)
        print(f"Expected SHA-256 : {EXPECTED_SHA256}", flush=True)

        if file_hash == EXPECTED_SHA256:
            print("[SUCCESS] Checkpoint verified! Integrity matches frozen training state.", flush=True)
            return True
        else:
            print(f"[FATAL ERROR] Checksum mismatch! Checkpoint is corrupt or modified.", flush=True)
            # Remove corrupted checkpoint to prevent loading corrupt model
            os.remove(CHECKPOINT_PATH)
            print(f"Removed corrupt checkpoint: {CHECKPOINT_PATH}", flush=True)

    download_url = os.environ.get("CHECKPOINT_DOWNLOAD_URL")
    if not download_url:
        print("[ERROR] Checkpoint missing and CHECKPOINT_DOWNLOAD_URL is not set.", flush=True)
        print("Please provide a valid download URL via environment variable or place best_model.pth manually.", flush=True)
        print(f"Expected location: {CHECKPOINT_PATH}", flush=True)
        return False

    print(f"Downloading checkpoint from: {download_url}...", flush=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    temp_path = CHECKPOINT_PATH + ".download"

    try:
        # Download with timeout and user-agent
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "AIOsteoporosisPrediction/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        downloaded_hash = compute_sha256(temp_path)
        print(f"Downloaded SHA-256: {downloaded_hash}", flush=True)
        print(f"Expected SHA-256  : {EXPECTED_SHA256}", flush=True)

        if downloaded_hash != EXPECTED_SHA256:
            os.remove(temp_path)
            raise ValueError(f"Downloaded file SHA-256 ({downloaded_hash}) does not match expected ({EXPECTED_SHA256})")

        os.replace(temp_path, CHECKPOINT_PATH)
        print("[SUCCESS] Download and SHA-256 verification complete!", flush=True)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[FAILED] Checkpoint download failed: {e}", flush=True)
        return False


if __name__ == "__main__":
    success = verify_or_download_checkpoint()
    sys.exit(0 if success else 1)
