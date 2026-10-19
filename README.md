# CASCADE — Clinical Anomalies & Scan Denoising Engine 🩺

> **Research Prototype Disclaimer**: CASCADE is an experimental medical imaging deep-learning and workflow visualization prototype. It is developed strictly for algorithmic research, architectural exploration, and academic demonstration. It is **not** an FDA-cleared or CE-marked medical device, does not diagnose pathology or assess emergency triage, and must not be used for primary clinical interpretation or patient care decisions.

---

## 1. Project Overview

**CASCADE (Clinical Anomalies & Scan Denoising Engine)** explores the intersection of deep-learning computer vision, explainable artificial intelligence (XAI), and multi-agent workflow orchestration in computed tomography (CT) imaging.

In diagnostic radiology, ultra-low-dose CT (ULDCT) protocols significantly reduce patient ionizing radiation exposure, but introduce quantum mottle, Poisson-Gaussian noise, and streak artifacts. These visual degradations obscure delicate anatomical boundaries and complicate radiologist interpretation.

CASCADE provides an end-to-end local platform to investigate:
1. **Neural Image Denoising**: Symmetric convolutional autoencoders (U-Net) trained to reconstruct structural detail from noisy low-dose inputs.
2. **Multi-Task Diagnostic Feature Extraction**: Backbone feature representations (ResNet50V2) for organ classification and candidate region-of-interest (ROI) localization.
3. **Explainable AI (XAI)**: Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize saliency maps corresponding to organ classification decisions.
4. **Multi-Agent Protocol Synthesis**: Stateful agent graphs (LangGraph) querying evidence-based clinical reference guidelines and formatting structured consultation reports.
5. **Zero-Trust Human-in-the-Loop (HITL) Workflow**: Transparent UI interfaces separating automated neural outputs from unauthenticated demo review acknowledgements.

---

## 2. Core Features

* **U-Net Grayscale CT Denoising**: Convolutional autoencoder architecture with skip connections trained on $256 \times 256$ grayscale CT slices, utilizing a sigmoid output layer mapping pixel intensities strictly to $[0.0, 1.0]$.
* **Independent Organ Classification**: Multi-task ResNet50V2 classifier predicting anatomical region across three primary target classes: `Chest`, `Brain`, and `Abdomen`.
* **Uncalibrated Model Scores**: Raw neural softmax probabilities are explicitly labeled as *Uncalibrated Model Scores* rather than certified clinical confidence.
* **Organ Discrepancy Warnings**: Real-time cross-validation alert flags when operator-selected organ presets conflict with deep learning model predictions.
* **Candidate ROI Bounding Boxes**: Regression-head coordinates normalized to $[0.0, 1.0]$ highlighting high-attenuation candidate regions for exploratory review without claiming certified pathology detection.
* **Organ-Classification Grad-CAM Saliency**: Heatmap overlay visualizing convolutional feature maps corresponding to organ class prediction (explicitly documented as organ saliency, not disease localization).
* **Scientific Metric Integrity**: Ground-truth Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) are calculated **only** when paired reference scans are provided; unpaired single scans truthfully display `N/A (no reference scan)`.
* **LangGraph Multi-Agent Medical Board**: Stateful 4-node agent orchestration:
  1. *Radiologist Vision Agent*: Parses U-Net reconstruction, organ predictions, and candidate ROIs.
  2. *Clinical RAG Specialist Agent*: Queries evidence-based reference guidelines via dense vector cosine similarity (Pinecone cloud or local fallback matrix).
  3. *Safety Auditor Agent*: Enforces non-diagnostic research notices and neutralizes emergency triage flags.
  4. *Human Review Node*: Manages review state and records demo acknowledgements.
* **Truthful Demo Review Acknowledgement**: Unauthenticated user clicks record `review_acknowledged: true` while strictly preserving `is_human_verified: false` and `reviewer_identity_verified: false` on the server. Zero doctor identities or degrees are fabricated.
* **Basic DICOM Metadata Sanitization Safeguard**: Basic field-level tag substitution safeguard (research demo; not independently certified de-identification). Complete removal of identifying information has not been independently verified; the project must not claim HIPAA compliance or certified de-identification, and sanitization does not guarantee that pixels, free text, metadata, or derived artifacts contain zero identifying information.
* **S3 PACS Cloud Vault Pathway**: Enterprise archival route supporting exact-byte storage of restored scans, Grad-CAM overlays, difference maps, and sanitized report JSON with AES256 server-side encryption, payload validation, and safe offline fallbacks.
* **Privacy-Conscious Telemetry (`telemetry.py`)**: Operational observability engine stripping patient identifiers, hospital names, internal disk paths, and credentials before recording trace metrics.

---

## 3. System Architecture & Data Flow

```mermaid
graph TD
    subgraph Ingestion ["1. Scan Ingestion & Metadata Safeguards"]
        A[Raw DICOM / CT Scan] --> B[pydicom / PIL Parser]
        B --> C["Basic Metadata Sanitization Safeguard<br/>(Research Demo; Not Independently Certified)"]
        C --> D[Normalization: [0.0, 1.0] Tensor (256x256x1)]
    end

    subgraph NeuralPipeline ["2. Cascaded Deep Learning Core"]
        D --> E[U-Net Denoising Autoencoder]
        E -->|Restored Scan| F[Isolated Noise Difference Map]
        E -->|Restored Scan| G[ResNet50V2 Multi-Task Model]
        G --> H[Organ Softmax Prediction]
        G --> I[Candidate ROI Coordinates]
        G --> J[Grad-CAM Organ Saliency Heatmap]
    end

    subgraph AgentBoard ["3. Multi-Agent Consultation (LangGraph)"]
        H & I & E --> K[Radiologist Vision Agent]
        K --> L[Clinical RAG Specialist Agent]
        L --> M[Safety Auditor & Escalator Agent]
        M --> N[Human-in-the-Loop Review Node]
        N --> O[Structured Consensus Research Report]
    end

    subgraph UIandVault ["4. PACS Presentation & Cloud Archival"]
        E & F & H & I & J & O --> P[Interactive Web PACS Dashboard]
        P -->|User Action: Acknowledge| Q[Demo Review Acknowledgement (is_human_verified=False)]
        P -->|User Action: Archive| R[AWS S3 Cloud Vault (AES256 Encrypted / Offline Mode)]
    end
```

### Component Roles

* **Flask (`app.py`)**: REST API server hosting inference endpoints, static asset serving, S3 upload sanitation, and demo review controllers.
* **TensorFlow / Keras 3 (`keras`)**: Deep learning execution engine running pre-trained U-Net (`medical_ct_denoiser_full.keras`) and ResNet50V2 (`medical_ct_diagnostic_model.keras`).
* **OpenCV (`cv2`) & Pillow (`PIL`)**: Color mapping, bilinear interpolation, bounding box rendering, and base64 image serialization.
* **PyDicom (`pydicom`)**: Medical CT file parsing, metadata extraction, and Hounsfield Unit (HU) min-max array windowing.
* **LangGraph & LangChain**: Stateful graph execution coordinating multi-node reasoning across vision observation, vector reference retrieval, and safety auditing.
* **Boto3 (`boto3`)**: AWS S3 integration operating in safe offline fallback mode when cloud credentials are unconfigured.

---

## 4. Model Artifact Specifications

> ⚠️ **Important Version Control Notice**: All `.keras` binary weight files are intentionally excluded from Git tracking via `.gitignore`. A fresh `git clone` contains the complete source code and test suite, but requires authorized model weight files to be placed in the project root before running full neural inference.

See [**`MODEL_ARTIFACTS.md`**](MODEL_ARTIFACTS.md) for complete model specifications.

| Model Name | Expected Root Filename | Size (Bytes) | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net CT Denoiser** | `medical_ct_denoiser_full.keras` | `23,388,275` | 22.30 MB | `ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842` |
| **Diagnostic ResNet50V2** | `medical_ct_diagnostic_model.keras` | `256,877,707` | 244.98 MB | `a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41` |

### Input/Output Tensor Contracts
* **U-Net CT Denoiser**:
  * Input: `(batch_size, 256, 256, 1)`, float32, range $[0.0, 1.0]$.
  * Output: `(batch_size, 256, 256, 1)`, float32, range $[0.0, 1.0]$ via `sigmoid` activation.
* **Diagnostic ResNet50V2**:
  * Input: `(batch_size, 256, 256, 1)`, float32, range $[0.0, 1.0]$.
  * Output 0: `(batch_size, 3)`, float32 softmax probabilities (`Chest`, `Brain`, `Abdomen`).
  * Output 1: `(batch_size, 4)`, float32 normalized bounding box `[ymin, xmin, ymax, xmax]`.

---

## 5. Local Setup & Quick Start Guide

### Prerequisites
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS.
* **Python**: Python `3.10` or `3.11` (tested on Python `3.11.9`).
* **Hardware**: Standard multi-core CPU (8 GB+ RAM recommended; GPU is optional).

### Step 1: Clone Repository & Create Virtual Environment

```powershell
# In Windows PowerShell:
git clone -b feat/enterprise-baseline https://github.com/ANURAGVIVEK0919/CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-.git
cd CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(On Linux / macOS bash, use: `source venv/bin/activate`)*

### Step 2: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Supply Authorized Model Files
Obtain the verified `.keras` model binaries from your organization's authorized artifact repository and place them in the project root:
* `./medical_ct_denoiser_full.keras`
* `./medical_ct_diagnostic_model.keras`

Verify file integrity using SHA-256 in PowerShell:
```powershell
Get-FileHash medical_ct_denoiser_full.keras, medical_ct_diagnostic_model.keras -Algorithm SHA256
```

### Step 4: Run the Regression Test Suite
Execute the project-local end-to-end integration test suite before launching the application:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_integration -v
```

All 28 tests must pass with exit code `0`. Tests execute 100% offline using synthetic fixtures and mock cloud clients.

### Step 5: Start the Local Web Application
Run the local development research-demo server:

```powershell
python app.py
```

* **Explicit Loopback Binding**: When started via `python app.py`, the application explicitly calls `app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)`. It binds exclusively to the local loopback interface (`127.0.0.1:5000`) and does not listen on external network interfaces (`0.0.0.0`).
* **Hardened Debug & Reloader Configuration**: Flask debug mode (`debug=False`) and the automatic reloader (`use_reloader=False`) are disabled to prevent arbitrary code execution vulnerabilities and suppress unwanted child worker thread spawning.
* **Research Demo Scope**: This local development startup is intended solely for local research exploration and demonstration. It must not be described or deployed as a clinical or production server.
* **Container vs. Local Startup**: In contrast, containerized deployments (`Dockerfile`) execute Gunicorn WSGI (`CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app`) to bind across all network interfaces within the container and listen on `$PORT` (default `7860`).
* **Local Browser URL**: Open [**`http://127.0.0.1:5000`**](http://127.0.0.1:5000) in your browser.
* **Safe Shutdown**: Press `Ctrl + C` in the terminal to terminate the process cleanly.

---

## 6. API Reference

| Endpoint | Method | Purpose | Demo / Offline Behavior |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | Renders the interactive dark PACS web dashboard. | Serves local HTML/CSS/JS with zero external CDN dependencies. |
| `/process` | `POST` | Processes batch uploaded CT slices (`.png`, `.jpg`, `.dcm`). Executes U-Net denoising, organ prediction, candidate ROI extraction, Grad-CAM generation, and multi-agent consultation. | Returns complete JSON results including base64 image data. Evaluates PSNR/SSIM only when clean pairs exist. |
| `/api/copilot/chat` | `POST` | Interacts with the LangGraph multi-agent board with custom clinical queries and scan context. | Queries local dense vector cosine matrix when Pinecone API key is unconfigured. |
| `/api/copilot/approve` | `POST` | Records a demo review acknowledgement for a processed scan slice. | Sets `review_acknowledged: true`. Strictly enforces `is_human_verified: false` and `reviewer_identity_verified: false`. Ignores client-supplied doctor names. |
| `/api/s3/upload` | `POST` | Archives a specific scan artifact (`restored_image`, `gradcam_overlay`, `diff_map`, `report_json`) to S3. | Truthfully returns `status: "Unavailable"` in local offline mode without falsifying a successful upload. Enforces AES256 encryption. |
| `/api/pipeline/run` | `POST` | Triggers batch DICOM ETL ingestion pipeline over local directory files. | Basic DICOM metadata sanitization safeguard (research demo; not independently certified de-identification); generates Data Lake manifest `pipeline_results.json`. |

---

## 7. Automated Testing & Verification Suite

The repository includes a comprehensive, project-local test suite in [`tests/test_integration.py`](tests/test_integration.py) covering 28 verified automated test gates across 6 categories:

1. **S3 Artifact Vault & Ingestion Integration Tests (11 tests)**:
   * Verifies exact-byte uploads for restored scans, Grad-CAM overlays, difference maps, and report JSON.
   * Enforces AES256 server-side encryption parameters.
   * Enforces strict allowlist filtering stripping patient identifiers, hospital names, and client paths from report JSON.
   * Rejects malformed image signatures, corrupted base64 payloads, and oversized payloads ($>25\text{ MB}$).
   * Verifies truthful `Unavailable` status when S3 is offline without simulating false success.
2. **U-Net Batch DICOM Pipeline Tests (4 tests)**:
   * Verifies pipeline output shapes, finite float values, and $[0.0, 1.0]$ normalization range.
   * Verifies aspect ratio and spatial resolution preservation across non-standard input geometries.
   * Verifies caching of a single U-Net model instance across slices in a batch.
   * Verifies explicit `FileNotFoundError` raised when required model weights are absent.
3. **Safe Operational Telemetry Tests (1 test)**:
   * Verifies regex sanitization stripping local drive paths (`D:\...`), AWS access keys (`AKIA...`), tokens, and PHI.
4. **Diagnostic Neural Inference & Grad-CAM Tests (3 tests)**:
   * Verifies candidate ROI coordinate bounding boxes are normalized and clamped to $[0.0, 1.0]$.
   * Verifies Grad-CAM generates valid overlays and handles model failures gracefully with clean `None` returns.
   * Verifies real local model smoke execution with independent organ classification across target classes.
5. **Multi-Agent Null-Safety & End-to-End Route Tests (4 tests)**:
   * Verifies agent handles `None` PSNR/SSIM metrics gracefully without crash or synthetic fallback.
   * Verifies end-to-end `/process` route execution using real model forward passes.
   * Verifies invalid or empty multipart uploads fail safely with HTTP `400`.
6. **HITL Review Integrity & Safety Tests (5 tests)**:
   * Verifies default scan state has `is_human_verified: false` and unreviewed status.
   * Verifies anonymous demo acknowledgement sets `review_acknowledged: true` but **never** `is_human_verified: true`.
   * Verifies client-supplied doctor names, degrees, or verification flags cannot forge verified clinician status.
   * Verifies S3 upload sanitizes forged doctor names and actively clamps `is_human_verified: false` in stored JSON.
   * Verifies review status does not bleed between separate scans.

---

## 8. Limitations & Clinical Safety Disclaimers

* **No Disease-Specific Pathology Classifier**: The multi-task network classifies 3 anatomical organ regions (`Chest`, `Brain`, `Abdomen`). It does **not** classify specific disease entities (such as ischemic stroke vs. intracerebral hemorrhage, pulmonary consolidation vs. pneumothorax).
* **Emergency Status Not Assessed**: Emergency triage cannot be evaluated based on organ classification or bounding box heuristics. `is_emergency` remains strictly `None` and `emergency_status` remains `"Not Assessed (No Certified Pathology Classifier)"`.
* **Grad-CAM Saliency Scope**: Grad-CAM attention maps visualize convolutional feature activations contributing to organ classification decisions. They do **not** represent verified pathology localization or surgical margins.
* **No Calibrated Clinical Probabilities**: Softmax output scores represent uncalibrated neural activations and must not be interpreted as validated statistical probabilities of disease presence.
* **Zero Authenticated Identity Verification**: The platform does not incorporate SAML SSO, OAuth2/OIDC, or hospital Active Directory integration to authenticate licensed medical staff. Real clinical sign-offs are explicitly unsupported.
* **Transient Review State**: Review acknowledgements are stored transiently in memory and browser session state. No append-only, non-repudiable audit database exists.
* **Not Independently Certified De-Identification**: The DICOM pipeline provides basic metadata sanitization safeguards (e.g., substituting patient names and hashing IDs). Complete removal of identifying information has not been independently verified, and the project must not claim HIPAA compliance or certified de-identification. Sanitization does not guarantee that image pixels (such as burned-in patient annotations or text), unstructured free text, private DICOM metadata tags, or derived artifacts contain zero identifying information. Real patient scans must not be uploaded without certified institutional de-identification.
* **Offline Cloud Vault**: S3 uploads operate in local offline mode by default. Live AWS cloud infrastructure has not been certified for protected health information (PHI) storage.

---

## 9. Developer Onboarding Handoff Checklist

Follow this checklist when setting up a fresh development environment:

1. [ ] **Clone Branch**: Ensure you are on branch `feat/enterprise-baseline`.
2. [ ] **Model Placement**: Place `medical_ct_denoiser_full.keras` and `medical_ct_diagnostic_model.keras` in the project root.
3. [ ] **Checksum Audit**: Run `Get-FileHash *.keras -Algorithm SHA256` and confirm matching digests in `MODEL_ARTIFACTS.md`.
4. [ ] **Environment Setup**: Create a clean Python 3.11 virtual environment (`python -m venv venv`).
5. [ ] **Dependency Installation**: Run `pip install -r requirements.txt`.
6. [ ] **Test Gate**: Run `python -m unittest tests.test_integration -v` and confirm 28/28 tests pass.
7. [ ] **Local Launch**: Run `python app.py` and open `http://127.0.0.1:5000`.
8. [ ] **Synthetic Test Scan**: Upload a non-identifying synthetic test image (ensuring no real patient data is used) to verify denoising, candidate ROI, and demo review acknowledgement.
9. [ ] **Offline Cloud Verification**: Click "Archive Restored Scan to S3" and confirm the system reports `Status: Unavailable (Offline Mode)`.

---

## 10. Future Work & Roadmap

### Research & Algorithmic Extensions
* Integration of patient-wise cross-validation splits to prevent slice leakage across training folds.
* Investigation of conditional generative diffusion models for high-frequency perceptual texture restoration.
* Training certified disease-specific classification heads on large public benchmark datasets (e.g., RSNA Intracranial Hemorrhage, NIH ChestX-ray14).

### Production & Clinical Prerequisites (Required Prior to Real-World Use)
* Formal FDA 510(k) or CE-MDR software-as-a-medical-device (SaMD) regulatory clearance.
* Enterprise Single Sign-On (SAML 2.0 / SMART-on-FHIR) with cryptographic digital signatures and role-based access control (RBAC).
* Independent validation and compliance certification of DICOM de-identification pipelines (e.g., full DICOM PS 3.15 Annex E de-identification profiles and burned-in OCR pixel scrubbing) under HIPAA Safe Harbor or Expert Determination standards.
* Cryptographic, immutable, append-only clinical audit trail (e.g., AWS QLDB or signed PostgreSQL ledger) compliant with 21 CFR Part 11 and HIPAA Security Rules.
* Certified DICOM DIMSE C-STORE / C-FIND networking node integration for direct PACS routing.
