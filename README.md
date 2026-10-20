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

> ⚠️ **Important Version Control Notice**: All `.keras` binary weight files are intentionally excluded from Git tracking via `.gitignore`. A fresh `git clone` contains the complete source code and test suite, but requires authorized model weight files to be placed in their designated paths before running full neural inference.

See [**`MODEL_ARTIFACTS.md`**](MODEL_ARTIFACTS.md) and [**`docs/model_dataset_registry.md`**](docs/model_dataset_registry.md) for complete specifications.

| Model Name | Expected Path | Size (Bytes) | Size (MB) | SHA-256 Checksum | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **U-Net CT Denoiser** | `medical_ct_denoiser_full.keras` | `23,388,275` | 22.30 MB | `ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842` | Legacy Baseline |
| **Diagnostic ResNet50V2** | `medical_ct_diagnostic_model.keras` | `256,877,707` | 244.98 MB | `a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41` | Exploratory Classifier |
| **Residual U-Net v3** | `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras` | `1,936,708` | 1.85 MB | `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea` | Preliminary Research Candidate (Frozen) |

### Input/Output Tensor Contracts
* **Legacy U-Net CT Denoiser**:
  * Input: `(batch_size, 256, 256, 1)`, float32, normalized to $[0.0, 1.0]$.
  * Output: `(batch_size, 256, 256, 1)`, float32, range $[0.0, 1.0]$ via `sigmoid` activation.
* **Diagnostic ResNet50V2**:
  * Input: `(batch_size, 256, 256, 1)`, float32, normalized to $[0.0, 1.0]$.
  * Output 0: `(batch_size, 3)`, float32 softmax probabilities (`Chest`, `Brain`, `Abdomen`).
  * Output 1: `(batch_size, 4)`, float32 normalized bounding box `[ymin, xmin, ymax, xmax]`.
* **Residual U-Net v3 (Preliminary Research Candidate)**:
  * Input: Native $512 \times 512 \times 1$ calibrated CT slice, linearly mapped from $[-1000.0, +1000.0]$ HU to $[0.0, 1.0]$.
  * Inference: Extracted into overlapping $128 \times 128 \times 1$ grid patches (stride 64) and blended across seams with a smooth 2D Hann window.
  * Output: Native $512 \times 512 \times 1$ tensor, domain-bounded clamped to $[0.0, 1.0]$, and inversely converted to $[-1000.0, +1000.0]$ HU.

---

## 5. Experimental Research Denoising Pipeline (Residual U-Net v3)

CASCADE provides a dedicated, hardened research endpoint (`POST /api/research/denoise_dicom`) implementing physical Hounsfield Unit (HU) calibrated denoising for medical CT scans using the **Residual U-Net v3 preliminary research candidate**.

### 5.1 Architecture & Native Patch Blending
* **Identity-Preserving Residual Learning**: Integrates a zero-initialized residual scaling layer ensuring that prior to training, the network computes an exact identity mapping $\hat{y} = x$.
* **Edge-Preserving Loss Objective**: Trained with a compound loss combining Huber intensity error with an L1 vector gradient regularizer ($\lambda_{\text{edge}} = 0.01$) to penalize anatomical edge blurring.
* **Seam-Free 2D Hann Windowing**: Native $512 \times 512$ scans are reconstructed from overlapping $128 \times 128$ patches weighted by separable 2D Hann windows, eliminating grid boundary discontinuities.
* **Domain-Bounded Clamping**: Enforces output clamping to $[0.0, 1.0]$ ($[-1000, +1000]$ HU) to eliminate unphysical air extrapolation while logging unclipped pre-clamp statistics.

### 5.2 Preliminary Benchmark Reconciliation (Locked 441-Slice Cohort)
Evaluated on a locked held-out test cohort of **441 matched CT slice pairs** across 3 patients (`L179`, `L210`, `L057`) from the TCIA Low-Dose CT Grand Challenge (Siemens SOMATOM Definition AS+ simulated reconstructions):

| Evaluation Metric | Raw Low-Dose CT | Residual U-Net v3 | Net Improvement | Measurement Conditions |
| :--- | :---: | :---: | :---: | :--- |
| **Soft-Tissue MAE (HU)** | 13.76 | **8.89** | **-4.87 HU (-35.39%)** | Abdominal soft-tissue $[-160, +240]$ HU; body contour $> -500$ HU |
| **Consistent Soft-Tissue PSNR (dB)** | 27.25 | **30.89** | **+3.64 dB** | Dynamic range $\text{MAX}_I = 400.0$ HU ($[-160, +240]$ HU) |
| **Historical Mixed-Domain PSNR (dB)** | 41.22 | **44.87** | **+3.65 dB** | Full clinical range $\text{MAX}_I = 2000.0$ HU ($+13.98$ dB mathematical offset) |
| **Consistent Soft-Tissue SSIM** | 0.8776 | **0.9250** | **+0.0474** | Soft-tissue window $[-160, +240]$ HU; peak range 400.0 HU |
| **Soft-Tissue Vector Gradient Error** | 93.37 | **74.39** | **-18.98 HU/px (-20.33%)** | Directional Sobel vector difference $\|\nabla I_{\text{pred}} - \nabla I_{\text{ref}}\|_2$ |

#### Important Metric & Acceptance Distinctions:
1. **Physical Soft-Tissue Window vs. Dynamic Range**: The clinical abdominal soft-tissue window is strictly physical $[-160.0, +240.0]$ HU (span width $W = 400.0$ HU) restricted to voxels within the patient body contour ($\text{HU} > -500$). The $[0.0, 400.0]$ domain represents the shifted/normalized metric computation interval ($I - (-160.0)$), not a physical HU window.
2. **PSNR Domain Distinction**: Consistent Soft-Tissue PSNR (30.89 dB, 400 HU range) must not be conflated with historical mixed-domain PSNR (44.87 dB, 2000 HU range). Because $\text{PSNR}_{\text{mixed}} = \text{PSNR}_{\text{consistent}} + 20 \log_{10}(5) = \text{PSNR}_{\text{consistent}} + 13.98\ \text{dB}$, they represent different dynamic ranges and are not directly interchangeable.
3. **Acceptance Criteria Codification & Temporal Provenance**: The engineering acceptance criteria codified in executable evaluation code (`evaluate_residual_unet_v3.py:L226-228`) are:
   * **Criterion 1 (Intensity)**: Soft-Tissue MAE $\le 10.0$ HU and improvement $\ge 0.50$ HU over raw low-dose CT ($\Delta \le -0.50$ HU).
   * **Criterion 2 (Structural Integrity)**: Full SSIM $\ge 0.9000$ and degradation $\le 0.015$ relative to raw scan ($\Delta \ge -0.015$).
   * **Criterion 3 (Edge Preservation)**: Vector gradient error $\le$ raw low-dose error.
   * *Temporal Provenance Disclosure*: While these criteria are codified in the test runner, because the runner and test execution occurred within the same working session without an immutable prior git commit, their pre-test timing is formally classified as `NOT_INDEPENDENTLY_VERIFIED`.
   * *Retrospective Variants & Disclaimers*: Thresholds such as "$\ge 20\%$ relative MAE reduction", "PSNR $\ge 42$ dB", or "SSIM $\ge 0.95$" were retrospective post-hoc observations or baseline comparisons; they were **not** prespecified criteria.
4. **Full-Scan vs. Soft-Tissue SSIM**: Reported Full SSIM (0.9857 vs. raw 0.9641) is computed over the full unmasked $512 \times 512$ canvas normalized across $[-1000, +1000]$ HU, where uniform zero-valued ambient air inflates both numbers; it is therefore excluded from primary claims in favor of the consistent soft-tissue windowed SSIM (0.9250 vs. raw 0.8776 across $[-160, +240]$ HU).
5. **Research Scope Notice**: This evaluation represents a preliminary retrospective benchmark across 3 patients and 441 slices. It does not constitute multi-vendor physical rescan validation or clinical approval.

### 5.3 Research Endpoint Privacy & Data Handling
* **In-Memory Ephemeral Execution**: Scans uploaded to `POST /api/research/denoise_dicom` are streamed into memory (`io.BytesIO`) under a strict 32 MB bounded limit. No input scans or derived DICOM files are stored on disk.
* **Selected Metadata Redaction**: Whitelist filtering returns only non-identifying physical CT calibration (`RescaleSlope`, `RescaleIntercept`) and spatial geometry (`PixelSpacing`, `SliceThickness`). Direct patient identifiers (`PatientName`, `PatientID`, `PatientBirthDate`), physician names, institution tags, and private vendor tags are completely excluded from JSON payloads.
* **Privacy Disclaimers**: Formal DICOM de-identification certification (e.g., DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile) is **not** claimed. Burned-in pixel annotations present on raw scans are not scrubbed. Blanket claims of "zero PHI" or "HIPAA compliance" are avoided.

---

## 6. Local Setup & Quick Start Guide

### Prerequisites
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS.
* **Python**: Python `3.10` or `3.11` (tested on Python `3.11.9`).
* **Hardware**: Standard multi-core CPU (8 GB+ RAM recommended; GPU is optional).

### Step 1: Clone Repository & Create Virtual Environment

```powershell
# In Windows PowerShell:
git clone https://github.com/ANURAGVIVEK0919/CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-.git
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
Obtain the verified `.keras` model binaries from your organization's authorized artifact repository and place them in their designated locations:
* `./medical_ct_denoiser_full.keras` (Legacy U-Net Baseline)
* `./medical_ct_diagnostic_model.keras` (Diagnostic ResNet50V2)
* `./artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras` (Residual U-Net v3)

Verify file integrity using SHA-256 in PowerShell:
```powershell
Get-FileHash medical_ct_denoiser_full.keras, medical_ct_diagnostic_model.keras -Algorithm SHA256
Get-FileHash artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras -Algorithm SHA256
```

### Step 4: Run the Full Offline Test Suite
Execute the project-local offline verification suite before launching the application:

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

All 126 tests must pass with exit code `0`. Tests execute 100% offline using synthetic fixtures and mock cloud clients.

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

### Planned Free Hosting Architecture (Hugging Face Static Frontend + OCI ARM64 CPU Backend)
* **Dual-Tier Zero-Subscription Model**:
  * **Frontend**: Hugging Face Static Space (`sdk: static`) preserving the canonical Space URL at [**`https://huggingface.co/spaces/ANURAG1909/cascade`**](https://huggingface.co/spaces/ANURAG1909/cascade). Operating as an `sdk: static` Space incurs zero platform compute charges and permanently removes Docker build errors, ZeroGPU constraints, and billing holds on Hugging Face.
  * **Backend**: Self-hosted CPU-only ARM64 backend on an Oracle Cloud Infrastructure (OCI) Always Free Ampere A1 VM (`VM.Standard.A1.Flex`, Ubuntu 22.04/24.04 ARM64) running Gunicorn behind an Nginx reverse proxy with Let's Encrypt SSL/TLS.
* **Pre-Deployment Readiness State**: **Neither the OCI backend nor the static frontend has been deployed yet.** This repository contains the complete, tested preparation packages under [`deploy/free_cpu_backend/`](deploy/free_cpu_backend/) and [`deploy/hf_static_frontend/`](deploy/hf_static_frontend/). Both environments remain unprovisioned pending manual administrative setup.
* **Model Artifact Integrity Gate**:
  * Pre-flight verification scripts ([`verify_backend_models.py`](deploy/free_cpu_backend/verify_backend_models.py)) strictly validate exact byte sizes and SHA-256 digests for all three models before starting application services:
    - `medical_ct_denoiser_full.keras`: `23,388,275` bytes | `ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842`
    - `medical_ct_diagnostic_model.keras`: `256,877,707` bytes | `a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41`
    - `candidate_unet_residual_edge_v3.keras`: `1,936,708` bytes (~1.85 MiB) | `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
* **OCI Always Free Eligibility & Policy Constraints**:
  * OCI Always Free resource eligibility, identity verification (credit card authorization), and host capacity in the account's designated Home Region must be verified manually by the tenancy owner before launch.
  * Always Free compute instances are subject to Oracle's idle-instance reclamation policy (<20% CPU, memory, and network utilization over a rolling 7-day period). Continuous 100% uninterrupted uptime is not guaranteed.
* **Regulatory & Clinical Safety Status**: CASCADE is an exploratory engineering prototype and research benchmark platform. It is **not** an FDA-cleared, CE-marked, or certified medical device (SaMD), and must not be used for clinical diagnosis, patient triage, or direct care delivery.

---

## 7. API Reference

| Endpoint | Method | Purpose | Demo / Offline Behavior |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | Renders the interactive dark PACS web dashboard. | Serves local HTML/CSS/JS with zero external CDN dependencies. |
| `/process` | `POST` | Processes batch uploaded CT slices (`.png`, `.jpg`, `.dcm`). Executes legacy U-Net denoising, organ prediction, candidate ROI extraction, Grad-CAM generation, and multi-agent consultation. | Returns complete JSON results including base64 image data. Evaluates PSNR/SSIM only when clean pairs exist. |
| `/api/research/denoise_dicom` | `POST` | Dedicated research endpoint for physical CT DICOM slices using Residual U-Net v3. | Strictly in-memory ephemeral execution, selected metadata whitelist redaction, zero disk save, base64 display previews, and dynamic output range accounting. |
| `/metrics` | `GET` | Prometheus & system health monitoring observability endpoint. | Exposes separate pipeline telemetry for legacy web pipeline vs. research DICOM pipeline, model availability, and service health. |
| `/api/copilot/chat` | `POST` | Interacts with the LangGraph multi-agent board with custom clinical queries and scan context. | Queries local dense vector cosine matrix when Pinecone API key is unconfigured. |
| `/api/copilot/approve` | `POST` | Records a demo review acknowledgement for a processed scan slice. | Sets `review_acknowledged: true`. Strictly enforces `is_human_verified: false` and `reviewer_identity_verified: false`. Ignores client-supplied doctor names. |
| `/api/s3/upload` | `POST` | Archives a specific scan artifact (`restored_image`, `gradcam_overlay`, `diff_map`, `report_json`) to S3. | Truthfully returns `status: "Unavailable"` in local offline mode without falsifying a successful upload. Enforces AES256 encryption. |
| `/api/pipeline/run` | `POST` | Triggers batch DICOM ETL ingestion pipeline over local directory files. | Basic DICOM metadata sanitization safeguard (research demo; not independently certified de-identification); generates Data Lake manifest `pipeline_results.json`. |

---

## 8. Automated Testing & Verification Suite

The repository includes a comprehensive, project-local test suite comprising **126 automated offline tests** across 10 test modules:

1. **Security & Research Endpoint Isolation ([`tests/test_research_api_security.py`](tests/test_research_api_security.py), 16 tests)**:
   * Validates endpoint isolation, 32 MB upload limit bounds, non-CT input rejection, and ephemeral in-memory processing.
   * Enforces exclusion of direct patient identifiers (`PatientName`, `PatientID`, `PatientBirthDate`) from API responses.
2. **Upload Safety & Ephemeral Storage Hardening ([`tests/test_upload_safety_hardening.py`](tests/test_upload_safety_hardening.py), 9 tests)**:
   * Validates upload isolation outside `static/`, unlinking temp files in `try/finally` blocks on success and error, extension allowlisting, and dual-layer 413 limits.
3. **Research Reproducibility Tooling ([`tests/test_research_reproducibility_tooling.py`](tests/test_research_reproducibility_tooling.py), 16 tests)**:
   * Validates patient-disjoint manifest splits, safe dry-run mode, overwrite locks, and metric aggregation with null handling for empty anatomical masks.
4. **End-to-End System Integration ([`tests/test_integration.py`](tests/test_integration.py), 28 tests)**:
   * Validates S3 cloud vault offline fallbacks, basic metadata sanitization safeguards, and review state integrity.
5. **RAG & Multi-Agent Quality ([`tests/test_model_rag_quality.py`](tests/test_model_rag_quality.py), 16 tests)**:
   * Validates clinical guidelines chunking, cosine fallback matrix, and LangGraph multi-agent board null-safety.
6. **Dataset Validation Tooling ([`tests/test_dataset_validation_tooling.py`](tests/test_dataset_validation_tooling.py), 12 tests)**:
   * Validates DICOM tag validation, integrity checks, and dataset verification tooling.
7. **Residual U-Net v3 Integration ([`tests/test_residual_unet_v3_integration.py`](tests/test_residual_unet_v3_integration.py), 7 tests)**:
   * Validates Flask API integration, multi-window display modes (`soft_tissue`, `lung`, `bone`, `full_body`), and spatial geometry preservation.
8. **Residual U-Net Architecture & Edge Regularization ([`tests/test_residual_unet_v3.py`](tests/test_residual_unet_v3.py), 7 tests)**:
   * Validates zero-initialized residual layer identity mapping and L1 vector gradient regularizer loss behavior.
9. **Physical HU Calibration Pipeline ([`tests/test_hu_calibrated_pipeline.py`](tests/test_hu_calibrated_pipeline.py), 8 tests)**:
   * Validates physical Hounsfield conversion, $[-1000, +1000]$ HU windowing, and anatomical mask extractions.
10. **Native Patch Reconstruction & Hann Blending ([`tests/test_residual_unet_native.py`](tests/test_residual_unet_native.py), 7 tests)**:
    * Validates 2D Hann window blending, grid patch extraction, and artifact-free seam elimination.
---

## 9. Limitations & Clinical Safety Disclaimers

* **No Disease-Specific Pathology Classifier**: The multi-task network classifies 3 anatomical organ regions (`Chest`, `Brain`, `Abdomen`). It does **not** classify specific disease entities (such as ischemic stroke vs. intracerebral hemorrhage, pulmonary consolidation vs. pneumothorax).
* **Emergency Status Not Assessed**: Emergency triage cannot be evaluated based on organ classification or bounding box heuristics. `is_emergency` remains strictly `None` and `emergency_status` remains `"Not Assessed (No Certified Pathology Classifier)"`.
* **Grad-CAM Saliency Scope**: Grad-CAM attention maps visualize convolutional feature activations contributing to organ classification decisions. They do **not** represent verified pathology localization or surgical margins.
* **No Calibrated Clinical Probabilities**: Softmax output scores represent uncalibrated neural activations and must not be interpreted as validated statistical probabilities of disease presence.
* **Zero Authenticated Identity Verification**: The platform does not incorporate SAML SSO, OAuth2/OIDC, or hospital Active Directory integration to authenticate licensed medical staff. Real clinical sign-offs are explicitly unsupported.
* **Transient Review State**: Review acknowledgements are stored transiently in memory and browser session state. No append-only, non-repudiable audit database exists.
* **Not Independently Certified De-Identification**: The DICOM pipeline provides basic metadata sanitization safeguards (e.g., substituting patient names and hashing IDs). Complete removal of identifying information has not been independently verified, and the project must not claim HIPAA compliance or certified de-identification. Sanitization does not guarantee that image pixels (such as burned-in patient annotations or text), unstructured free text, private DICOM metadata tags, or derived artifacts contain zero identifying information. Real patient scans must not be uploaded without certified institutional de-identification.
* **Offline Cloud Vault**: S3 uploads operate in local offline mode by default. Live AWS cloud infrastructure has not been certified for protected health information (PHI) storage.

---

## 10. Developer Onboarding Handoff Checklist

Follow this checklist when setting up a fresh development environment:

1. [ ] **Clone Branch**: Ensure you are on the canonical `main` branch (or your intended feature branch).
2. [ ] **Model Placement**: Place `medical_ct_denoiser_full.keras` and `medical_ct_diagnostic_model.keras` in the project root, and `candidate_unet_residual_edge_v3.keras` under `artifacts/experiments/unet_residual_edge_v3/run_001/`.
3. [ ] **Checksum Audit**: Run `Get-FileHash *.keras -Algorithm SHA256` and confirm matching digests in `MODEL_ARTIFACTS.md`.
4. [ ] **Environment Setup**: Create a clean Python 3.11 virtual environment (`python -m venv venv`).
5. [ ] **Dependency Installation**: Run `pip install -r requirements.txt`.
6. [ ] **Test Gate**: Run `python -m unittest discover tests` and confirm all 126 tests pass offline.
7. [ ] **Local Launch**: Run `python app.py` and open `http://127.0.0.1:5000`.
8. [ ] **Synthetic Test Scan**: Upload a non-identifying synthetic test image (ensuring no real patient data is used) to verify denoising, candidate ROI, and demo review acknowledgement.
9. [ ] **Offline Cloud Verification**: Click "Archive Restored Scan to S3" and confirm the system reports `Status: Unavailable (Offline Mode)`.

---

## 11. Future Work & Roadmap

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

---

## Portfolio Showcase & Rights Notice

CASCADE is published publicly as an engineering portfolio and
research-demo showcase. This repository is not released under
an open-source license.

Except for the limited rights provided through GitHub's Terms
of Service and public-repository functionality, no additional
license or permission is granted to use, reproduce, modify,
redistribute, sublicense, or create derivative works from the
original project materials without prior written authorization
from the applicable rights holder.

Rights in original project materials remain with their respective
rights holders. This notice does not claim ownership of third-party
software, fonts, images, datasets, models, or other assets. Those
materials remain subject to their respective licenses and terms.

Medical imaging datasets and trained model-weight files are
intentionally excluded from this repository. Their use and
redistribution may be subject to separate terms; publication of
this source code does not grant additional rights to those assets.

This project is a research and engineering demonstration, not a
certified clinical diagnostic system.
