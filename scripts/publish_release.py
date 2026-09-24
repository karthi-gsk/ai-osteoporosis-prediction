"""
Automated GitHub Release Publishing and Independent Download Verification Script.
Phase 8F — AI Osteoporosis Prediction System
"""

import os
import sys
import hashlib
import json
import subprocess
import urllib.request
import urllib.error

EXPECTED_SIZE = 134266309
EXPECTED_SHA256 = "6ccc18b8b71b4bd3d81740aeb9be48271ae24aef119fd0d37801362017aa8faa"
CHECKPOINT_PATH = os.path.join("models", "checkpoints", "resnet18", "best_model.pth")
REPO_OWNER = "karthi-gsk"
REPO_NAME = "ai-osteoporosis-prediction"
TAG_NAME = "v1.0.0"
RELEASE_TITLE = "ResNet18 Research Model — v1.0.0"

RELEASE_NOTES = """# ResNet18 Research Model — v1.0.0

This release provides the frozen ResNet18 model checkpoint for the **AI Osteoporosis Prediction System**, trained and evaluated exclusively for non-clinical academic research.

---

### Artifact Verification
* **Asset Filename:** `best_model.pth`
* **File Size:** `134,266,309` bytes (128.05 MiB)
* **SHA-256 Checksum:** `6ccc18b8b71b4bd3d81740aeb9be48271ae24aef119fd0d37801362017aa8faa`
* **Architecture:** ResNet18 (Torchvision backbone with custom 3-class linear head)
* **Training Checkpoint:** Epoch 21 (`best_val_f1 = 0.7913`)
* **Checkpoint Contents:** PyTorch state dictionary containing `model_state_dict`, `optimizer_state_dict`, `class_names`, and `training_config`.

---

### Held-Out Test Set Performance (n=135)
* **Test Accuracy:** 80.74% (109 / 135) [95% CI: 73.33%, 87.41%]
* **Macro Precision:** 0.8077
* **Macro Recall:** 0.8231
* **Macro F1-Score:** 0.8072 [95% CI: 73.35%, 86.95%]
* **Macro ROC-AUC:** 0.9174
* **Expected Calibration Error (ECE):** 0.0984

---

### Dataset Attribution & License
* **Dataset:** *Knee X-ray Osteoporosis Database*
* **Authors:** Insha Majeed Wani and Sakshi Arora
* **DOI:** [10.17632/fxjm8fb6mw.2](https://data.mendeley.com/datasets/fxjm8fb6mw/2) (Mendeley Data)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Attribution:** The original knee radiographs were compiled by Insha Majeed Wani and Sakshi Arora for deep learning research. Note that 76.5% of images in the source dataset lack explicit patient identifiers.

---

### Academic & Research Disclaimer
> [!IMPORTANT]
> This checkpoint is provided strictly for academic and scientific research purposes. It is **not a certified medical diagnostic device** and must not be used for clinical decision-making or establishing patient diagnoses. It does not measure physical Bone Mineral Density (BMD) or generate DXA T-scores.
"""


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def get_github_token() -> str:
    proc = subprocess.Popen(
        ["git", "credential", "fill"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, _ = proc.communicate("protocol=https\nhost=github.com\n\n")
    for line in stdout.splitlines():
        if line.startswith("password="):
            return line[9:].strip()
    raise RuntimeError("Could not retrieve GitHub token from Git credential manager.")


def main():
    print("=== TASK 1: VERIFY ARTIFACT ===", flush=True)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    actual_size = os.path.getsize(CHECKPOINT_PATH)
    actual_hash = compute_sha256(CHECKPOINT_PATH)

    print(f"File Size : {actual_size} bytes (Expected: {EXPECTED_SIZE})", flush=True)
    print(f"SHA-256   : {actual_hash}", flush=True)
    print(f"Expected  : {EXPECTED_SHA256}", flush=True)

    assert actual_size == EXPECTED_SIZE, "Size mismatch!"
    assert actual_hash == EXPECTED_SHA256, "SHA-256 mismatch!"
    print("[PASSED] Artifact integrity verified 100%!\n", flush=True)

    print("=== TASK 3: PUBLISH GITHUB RELEASE ===", flush=True)
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "AIOsteoporosisRelease/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Check if release already exists
    release_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{TAG_NAME}"
    req = urllib.request.Request(release_url, headers=headers)
    release_data = None
    try:
        with urllib.request.urlopen(req) as resp:
            release_data = json.loads(resp.read().decode())
            print(f"Existing release found: {release_data.get('html_url')}", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Creating new GitHub Release for tag {TAG_NAME}...", flush=True)
        else:
            raise

    if not release_data:
        create_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
        payload = {
            "tag_name": TAG_NAME,
            "name": RELEASE_TITLE,
            "body": RELEASE_NOTES,
            "draft": False,
            "prerelease": False,
        }
        create_req = urllib.request.Request(
            create_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(create_req) as resp:
            release_data = json.loads(resp.read().decode())
        print(f"Release created successfully! URL: {release_data.get('html_url')}", flush=True)

    # Check if asset is already attached
    assets = release_data.get("assets", [])
    existing_asset = next((a for a in assets if a.get("name") == "best_model.pth"), None)

    upload_url_template = release_data.get("upload_url", "")
    upload_base = upload_url_template.split("{")[0]

    if existing_asset:
        print(f"Asset 'best_model.pth' already exists on release. Browser download URL: {existing_asset.get('browser_download_url')}", flush=True)
        download_url = existing_asset.get("browser_download_url")
    else:
        print(f"Uploading 'best_model.pth' ({actual_size} bytes) to GitHub Release...", flush=True)
        upload_endpoint = f"{upload_base}?name=best_model.pth"
        
        with open(CHECKPOINT_PATH, "rb") as f:
            file_bytes = f.read()

        upload_req = urllib.request.Request(
            upload_endpoint,
            data=file_bytes,
            headers={
                **headers,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(file_bytes)),
            },
            method="POST",
        )
        with urllib.request.urlopen(upload_req) as resp:
            asset_data = json.loads(resp.read().decode())
        download_url = asset_data.get("browser_download_url")
        print(f"Asset upload complete! Download URL: {download_url}", flush=True)

    print("\n=== TASK 4: POST-PUBLICATION INDEPENDENT DOWNLOAD VERIFICATION ===", flush=True)
    test_download_path = "models/checkpoints/resnet18/temp_download_verify.pth"
    print(f"Downloading release asset from public URL: {download_url}...", flush=True)

    # Follow redirects for download
    req_dl = urllib.request.Request(download_url, headers={"User-Agent": "AIOsteoVerify/1.0"})
    with urllib.request.urlopen(req_dl) as resp_dl, open(test_download_path, "wb") as out_f:
        while chunk := resp_dl.read(1024 * 1024):
            out_f.write(chunk)

    dl_size = os.path.getsize(test_download_path)
    dl_hash = compute_sha256(test_download_path)

    print(f"Downloaded Size    : {dl_size} bytes", flush=True)
    print(f"Downloaded SHA-256 : {dl_hash}", flush=True)
    print(f"Expected SHA-256   : {EXPECTED_SHA256}", flush=True)

    # Clean up test download
    if os.path.exists(test_download_path):
        os.remove(test_download_path)

    assert dl_size == EXPECTED_SIZE, f"Downloaded size {dl_size} != expected {EXPECTED_SIZE}"
    assert dl_hash == EXPECTED_SHA256, f"Downloaded SHA-256 {dl_hash} != expected {EXPECTED_SHA256}"

    print("\n[SUCCESS] Release verified independently! Downloaded asset matches frozen checkpoint 100%!", flush=True)
    print(f"Published Release URL : {release_data.get('html_url')}", flush=True)
    print(f"Direct Asset URL      : {download_url}", flush=True)


if __name__ == "__main__":
    main()
