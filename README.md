# AI Osteoporosis Prediction System
### Explainable Deep Learning for Knee Radiograph Bone Health Assessment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CPU-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> **ACADEMIC & RESEARCH NOTICE**: This application is strictly an educational and academic research tool. It is **not a certified medical diagnostic device** (FDA/CE marked) and does not calculate clinical Dual-energy X-ray Absorptiometry (DXA) Bone Mineral Density (BMD) T-scores. It must never be used as a substitute for professional medical diagnosis or clinical judgment.

---

## 1. Project Overview

The **AI Osteoporosis Prediction System** is an end-to-end, explainable deep-learning research platform engineered to evaluate bone structural indicators from plain knee X-ray radiographs. Built on a frozen, benchmarked **ResNet18** convolutional backbone, the system classifies input radiographs into three diagnostic categories:
1. **Normal Bone Density**
2. **Osteopenia** (Mild-to-moderate bone density attenuation)
3. **Osteoporosis** (Severe bone mineral loss and microarchitectural deterioration)

Beyond raw categorization, the platform integrates **genuine Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually highlight the specific trabecular bone and joint space regions driving neural network inferences, and provides clinical-grade **PDF research report export**.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React 19 + Vite)"]
        UI["Upload Interface & Preview"] --> API_CLIENT["Frontend API Service (api.js)"]
        DASHBOARD["Interactive Analysis Dashboard"] --> CAM_VIEW["Grad-CAM Dynamic Viewer"]
        DASHBOARD --> PDF_EXP["jsPDF Report Generator (pdfReport.js)"]
    end

    subgraph Server["FastAPI Backend (Python 3.10+)"]
        ROUTER["API Router (/api/predict, /api/health)"]
        VALIDATOR["Pillow Image Validator (Dimensions, MIME, Integrity)"]
        AI_SVC["Inference Service (ai_service.py)"]
        ROUTER --> VALIDATOR
        VALIDATOR --> AI_SVC
    end

    subgraph Engine["AI & Explainability Engine"]
        PREPROC["Input Normalization (224x224, ImageNet Norm)"]
        RESNET["Frozen ResNet18 Backbone (Epoch 21 Checkpoint)"]
        GRADCAM["Grad-CAM Engine (model.layer4[-1] Hooks)"]
        AI_SVC --> PREPROC
        PREPROC --> RESNET
        RESNET --> GRADCAM
    end

    API_CLIENT -->|multipart/form-data| ROUTER
    GRADCAM -->|Overlay & Heatmap Data URIs| DASHBOARD
    RESNET -->|Softmax Probabilities| DASHBOARD
```

---

## 3. Technology Stack

* **Deep Learning & Explainability**:
  * PyTorch 2.6 (CPU-optimized inference runtime)
  * Torchvision 0.21
  * Custom thread-safe Grad-CAM hook manager (`scripts/gradcam.py`)
  * OpenCV & Pillow for colormap blending and tensor transformations
* **Backend Services**:
  * FastAPI 0.115 (Asynchronous REST API)
  * Uvicorn (ASGI production server)
  * Pydantic v2 (Strict request/response data contracts)
  * Dynamic CORS origin management
* **Frontend Web Application**:
  * React 19 + Vite 6
  * Modern Vanilla CSS design system (Tailwind-free, glassmorphic dark-mode palette, responsive layouts)
  * Lucide React icons
  * jsPDF with clean text wrapping, aspect-ratio preservation, and WHO DXA reference tables

---

## 4. Model Evaluation & Benchmark Results

The frozen ResNet18 model was selected after rigorous multi-model comparison across Custom CNN, EfficientNet-B0, DenseNet-121, and ResNet-18 architectures on a verified, strictly deduplicated knee radiograph dataset.

### Held-Out Test Evaluation (135 Unseen Radiographs)

| Evaluation Metric | Measured Value | 95% Confidence Interval |
| :--- | :--- | :--- |
| **Overall Accuracy** | **80.74%** (109 / 135) | [73.5%, 87.2%] |
| **Macro Precision** | **0.8099** | [0.738, 0.874] |
| **Macro Recall** | **0.8074** | [0.735, 0.872] |
| **Macro F1-Score** | **0.8072** | [0.735, 0.871] |
| **Macro ROC-AUC** | **0.9174** | [0.865, 0.958] |

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.8163 | 0.8889 | **0.8511** | 45 |
| **Osteopenia** | 0.7708 | 0.8222 | **0.7957** | 45 |
| **Osteoporosis** | 0.8421 | 0.7111 | **0.7711** | 45 |

### Confusion Matrix

```
                Predicted Normal   Predicted Osteopenia   Predicted Osteoporosis
Actual Normal          40                   5                      0
Actual Osteopenia       4                  37                      4
Actual Osteoporosis     5                   6                     32
```

---

## 5. Grad-CAM Explainability Methodology

The explainability engine implements genuine Gradient-weighted Class Activation Mapping:
1. **Target Feature Layer**: The final residual block `model.layer4[-1]` (BasicBlock 2, yielding 512 channels at $7 \times 7$ spatial resolution).
2. **Gradient Backpropagation**: Computes the gradients of the unnormalized target class logit $y^c$ with respect to feature activation maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Rectified Linear Combination**: Generates the coarse localization map by linearly combining activation maps weighted by $\alpha_k^c$ and applying ReLU to capture positive evidence:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Colormap Blending**: Normalizes activations to $[0, 1]$, resizes to $224 \times 224$ via bilinear interpolation, maps to the OpenCV `JET` colormap, and alpha-blends ($\alpha = 0.45$) over the input radiograph.

---

## 6. Local Installation & Setup

### Prerequisites
* **Python 3.10+** (64-bit)
* **Node.js 18+** and **npm**

### Step 1: Clone Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### Step 2: Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install CPU-only dependencies
pip install -r requirements-cpu.txt
```

### Step 3: Checkpoint Verification
Run the verification script to confirm the frozen ResNet18 checkpoint is present and valid:
```bash
python scripts/download_checkpoint.py
```
*Expected SHA-256:* `6ccc18b8b71b4bd3d81740aeb9be48271ae24aef119fd0d37801362017aa8faa`

### Step 4: Frontend Setup
```bash
cd ../frontend
npm install
```

### Step 5: Run Application Locally
* **Terminal 1 (Backend)**:
  ```bash
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  API Docs (Swagger): `http://127.0.0.1:8000/docs`
* **Terminal 2 (Frontend)**:
  ```bash
  cd frontend
  npm run dev
  ```
  Frontend Dashboard: `http://localhost:5173`

---

## 7. Production Docker Deployment

### Building and Running with Docker
```bash
# Build production Docker container
docker build -f backend/Dockerfile -t osteoporosis-ai-backend .

# Run container (mapping port 8000)
docker run -d -p 8000:8000 \
  -e PORT=8000 \
  -e CHECKPOINT_DOWNLOAD_URL="https://github.com/<USER>/<REPO>/releases/download/v1.0.0/best_model.pth" \
  --name osteoporosis-api osteoporosis-ai-backend
```

### Environment Variables
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port on which FastAPI Uvicorn server listens |
| `HOST` | `0.0.0.0` | Network binding host address |
| `MODEL_CHECKPOINT_PATH` | `/app/models/...` | Path to frozen `best_model.pth` |
| `CHECKPOINT_DOWNLOAD_URL` | *None* | Direct URL to download checkpoint if missing on container boot |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowed domains (e.g., `https://myapp.vercel.app`) |

---

## 8. Academic Research & Clinical Limitations

1. **Anatomical Scope**: This system was trained and evaluated **exclusively on plain knee X-ray radiographs**. It has not been validated on lumbar spine, proximal femur, or hip radiographs.
2. **Not a DXA Replacement**: Dual-energy X-ray Absorptiometry (DXA) remains the clinical gold standard for measuring areal Bone Mineral Density ($g/cm^2$) and establishing diagnostic T-scores. This CNN estimates visual radiograph features and does not measure true physical density.
3. **Probabilistic Outputs**: Softmax scores represent mathematical category distribution over the training manifold, not absolute calibrated probabilities of disease prevalence.
4. **Grad-CAM Interpretation**: Activation heatmaps denote neural network feature correlation. They do not constitute anatomical segmentations of osteoporotic lesions.
5. **Sample Size & Variance**: The held-out test evaluation was conducted on 135 strictly isolated knee radiographs. Clinical validation across diverse demographic populations and imaging hardware is required before any translational application.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
