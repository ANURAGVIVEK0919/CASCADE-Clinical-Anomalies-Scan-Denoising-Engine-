# CASCADE Model & Dataset Registry

This registry provides a rigorous, technical inventory of all model artifacts, candidate experiments, external foundation models, and interpretability methods present in the CASCADE repository.

---

## 1. Inventory Summary

| Identifier | Type | Architecture / Base | File / Artifact Path | Status | Retrainable Locally? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CT Denoiser (Full - Primary)** | Neural Restorer | 4-Level Residual U-Net | `medical_ct_denoiser_full.keras` | **VALIDATED** (+6.78 dB PSNR, +0.2232 SSIM on test set) | Retrained & Verified |
| **Candidate U-Net Denoiser** | Neural Restorer (Exp 1) | 4-Level Residual U-Net | `artifacts/experiments/unet_denoiser/run_001/candidate_unet_denoiser.keras` | **VALIDATED** (+5.44 dB PSNR, +0.2250 SSIM on test set) | Yes (CPU Verified) |
| **CT Denoiser (1-Epoch)** | Intermediate Checkpoint | 4-Level Residual U-Net | `medical_ct_denoiser_full_1epoch.keras`| Incomplete Checkpoint (Preserved) | Yes |
| **Diagnostic ResNet (Baseline)** | Multi-Task Classifier/BBox | ResNet50V2 + Dual Heads | `medical_ct_diagnostic_model.keras` | **COMPROMISED** (Near 0 IoU; Pelvis as Brain; retained for legacy fallback) | Preserved |
| **Candidate Lesion Localization** | Lesion ROI BBox Regressor | ResNet50V2 + Dense Head | `artifacts/experiments/lesion_localization/run_001/candidate_diagnostic_model.keras` | **PRELIMINARY** (Mean IoU 0.0236, 3.4x over baseline on 304 test slices) | Yes (CPU Verified) |
| **3-Region Classifier (Head/Chest/Abdomen)**| Anatomical Region Classifier| Multi-Class ResNet | N/A | **BLOCKED** (Pending authorized cranial CT dataset e.g. CQ500/RSNA) | Blocked |
| **RAG Embedder** | Dense Vector Embedder | MiniLM-L6-v2 (384-d) | `sentence-transformers/all-MiniLM-L6-v2` | **VALIDATED** (MRR 1.0000, Hit@1 100%, Abstention 50%) | Keep Pretrained |
| **Clinical Report LLM** | Generative Radiologist | Qwen2.5-72B-Instruct | HuggingFace Serverless Inference API | External Hosted API (Demonstration Only) | Cloud API Only |
| **Residual U-Net v3 (Candidate)** | Edge-Preserving Restorer | Native Patch Residual U-Net | `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras` | **PRELIMINARY RESEARCH CANDIDATE** (+3.64 dB ST-PSNR, -35.39% ST-MAE, -20.33% VecGrad on 441 fresh slices) | Checkpoint Frozen & Verified |
| **Grad-CAM Saliency** | Interpretability Method | Feature Attribution | `gradcam_utils.py` / `gradcam.py` | **VALIDATED** (Algorithmic Method, No weights) | N/A (Analytical) |

---


## 2. Detailed Model Specifications

### Model 1: U-Net CT Denoiser (Baseline & Candidate)
* **File Identifier**: `medical_ct_denoiser_full.keras` (23.4 MB) & `artifacts/experiments/unet_denoiser/run_001/candidate_unet_denoiser.keras` (23.4 MB)
* **Architecture**: 4-level encoder-decoder U-Net with skip connections and residual additions:
  * Input Layer: `(None, 256, 256, 1)` float32 normalized to `[0.0, 1.0]`.
  * Encoder Blocks: 4 convolutional stages with MaxPooling (`32 -> 64 -> 128 -> 256` filters).
  * Bottleneck: Convolutional block with `512` filters.
  * Decoder Blocks: 4 transposed convolution stages (`Conv2DTranspose` + concatenation + `Conv2D`).
  * Output Layer: `Conv2D(1, (1, 1), activation='sigmoid')` -> `(None, 256, 256, 1)`.
  * Parameter Count: 1,930,657 parameters (1,927,841 trainable).
* **Actual Task**: Image restoration / denoising of single-slice axial CT images.
* **Training & Evaluation Provenance**:
  * Evaluated on authentic NIH DeepLesion held-out test split: **304 slices across 39 independent patients** (zero patient leakage).
  * **Baseline Checkpoint (`medical_ct_denoiser_full.keras`)**:
    * Baseline Noisy PSNR: `32.78 dB` -> Restored PSNR: `39.56 dB` (**+6.78 dB gain**).
    * Baseline Noisy SSIM: `0.7110` -> Restored SSIM: `0.9342` (**+0.2232 gain**).
  * **Candidate Checkpoint (`run_001/candidate_unet_denoiser.keras`)**:
    * Trained reproducibly on 1,490 training pairs, 275 validation pairs with early stopping.
    * Baseline Noisy PSNR: `32.78 dB` -> Restored PSNR: `38.22 dB` (**+5.44 dB gain**).
    * Baseline Noisy SSIM: `0.7110` -> Restored SSIM: `0.9361` (**+0.2250 gain**).
  * **Acceptance Decision**: Baseline `medical_ct_denoiser_full.keras` retained as primary production weights due to superior PSNR restoration (+6.78 dB vs +5.44 dB). Candidate experiment saved with full logs and provenance.
* **Genuine LDCT Acquisition Status**:
  * Mayo Clinic TCIA Low Dose CT collection is >1.2 TB and requires signed TCIA Limited Access Agreement / NBIA Data Retriever. Status: **BLOCKED PENDING OWNER REGISTRATION**.
  * Controlled dual-component clinical Poisson-Gaussian noise reference pairs used for reproducible evaluation.

---

### Model 2: ResNet50V2 Lesion Localization Model
* **Baseline File**: `medical_ct_diagnostic_model.keras` (256.9 MB)
* **Candidate File**: `artifacts/experiments/lesion_localization/run_001/candidate_diagnostic_model.keras` (256.9 MB)
* **Architecture**: Transfer learning backbone (ResNet50V2 ImageNet pretrained) with dual task heads:
  * Input Layer: `(None, 256, 256, 1)` float32 in `[0.0, 1.0]`.
  * Channel Replication: Triplicated to 3 channels for ResNet50V2 ImageNet input compatibility.
  * Backbone: ResNet50V2 convolutional feature extractor (frozen ImageNet weights, 2048-dim GAP).
  * Localization Head: Dense(256, relu) -> BatchNorm -> Dropout(0.3) -> Dense(128, relu) -> BatchNorm -> Dropout(0.2) -> Dense(4, sigmoid) `[ymin, xmin, ymax, xmax]`.
  * Total Parameters: 24,082,151 parameters.
* **Actual Task**: Lesion ROI candidate localization (normalized bounding box regression) against authentic NIH DeepLesion RECIST annotations.
* **Evaluation Comparison (Held-Out Test Set: 304 slices, 39 patients)**:
  * Baseline Checkpoint (`medical_ct_diagnostic_model.keras`):
    * Mean IoU: `0.0068`
    * Median IoU: `0.0000`
    * Precision@IoU>=0.50: `0.0%`
    * Missed Detections (IoU < 0.10): `300 / 304 (98.7%)`
  * Candidate Checkpoint (`run_001/candidate_diagnostic_model.keras`):
    * Mean IoU: `0.0236` (**3.4x improvement over baseline**)
    * Precision@IoU>=0.50: `0.7%`
    * Precision@IoU>=0.25: `3.0%`
    * Missed Detections: `278 / 304 (91.5%)`
* **Status**: **PRELIMINARY / RESEARCH LOCALIZATION**.
  * DeepLesion RECIST annotations mark target primary measurement lesions; global feature pooling limits fine spatial localization compared to two-stage regional detector backbones (e.g., Faster R-CNN / RetinaNet).
  * Retained baseline in root for legacy demo interface compatibility; candidate preserved in `artifacts/experiments/`.

---

### Model 3: 3-Region Anatomical Classifier (Head vs. Chest vs. Abdomen)
* **Status**: **BLOCKED FROM TRAINING**.
* **Rationale**: NIH DeepLesion contains strictly chest, abdomen, pelvis, and extremity CT scans; it contains 0 cranial CT scans.
* The legacy codebase remapped DeepLesion Pelvis scans to "Brain CT", creating invalid ground truth.
* Per Phase 1.C requirements: "If you cannot obtain a sufficiently justified head/chest/abdomen classification dataset, mark this model BLOCKED instead of training on misleading labels."
* Training is suspended until authorized external cranial CT data (e.g. CQ500 or RSNA Intracranial Hemorrhage) is formally acquired under permitted non-commercial terms.

---

### Model 4: Semantic RAG Embedding Model
* **Identifier**: `sentence-transformers/all-MiniLM-L6-v2`
* **Architecture**: 6-layer BERT-based transformer encoder with mean pooling and L2 normalization:
  * Input: Tokenized clinical guideline text queries or structural chunks (max sequence length 256 tokens).
  * Output: 384-dimensional dense float vector (`norm == 1.0`).
* **Execution Mode & Cloud Connectivity**: Operates in **local offline mode** (using local semantic retrieval / cached knowledge chunks) unless live Pinecone Cloud connectivity (`PINECONE_API_KEY`) is separately verified.
* **Evaluation Metrics (Held-Out Query Suite)**:
  * In-Domain Mean Reciprocal Rank (MRR): **1.0000**
  * Hit@1: **100.0%**
  * Out-of-Domain Abstention Accuracy: **50.0%** (Score threshold: 0.40)
* **Retraining Decision**: **KEEP PRETRAINED**. Pretrained embeddings provide high semantic recall without risk of domain over-fitting.

---

### Model 5: Clinical Report Synthesis LLM
* **Identifier**: `Qwen/Qwen2.5-72B-Instruct`
* **Architecture**: 72-billion parameter auto-regressive Transformer.
* **Actual Task**: Synthesizing radiologist observations, U-Net metrics, and retrieved clinical guideline excerpts into structured impression summaries.
* **Execution Mode**: Remote inference via Hugging Face Serverless API (`HF_TOKEN`). Optional; fails gracefully to a non-diagnostic template summary if unconfigured.
* **Retraining Decision**: **DO NOT RETRAIN**. Cloud API demonstration component.

---

### Method 6: Grad-CAM Saliency Visualization
* **Identifier**: `gradcam_utils.py` / `gradcam.py`
* **Nature**: Post-hoc gradient-weighted class activation mapping (Selvaraju et al., 2017).
* **Architecture / Weights**: None. It is an algorithmic analytical tool, not a parameterized neural network.
* **Actual Task**: Computes gradients of target output with respect to convolutional feature maps of `conv5_block3_out` in ResNet50V2.

---

### Model 7: Residual U-Net v3 Native-Resolution CT Denoiser (Research Candidate)
* **File Identifier**: `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras` (139.5 MB)
* **Checkpoint SHA-256**: `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
* **Architecture & Preprocessing Configuration**:
  * **Model Architecture**: Deep Residual U-Net with skip connections and zero-initialized residual correction conv layer (`prediction = input + residual_correction`).
  * **Input Shape**: `(None, 128, 128, 1)` float32.
  * **Patch Extraction & Blending**: 2D grid patch extraction with stride 64 and seamless 2D Hann window overlapping reconstruction. Preserves full $512 \times 512$ native matrix resolution with zero downsampling.
  * **HU Calibration Preprocessing**: Fixed linear mapping $[-1000.0, +1000.0]$ HU $\rightarrow [0.0, 1.0]$.
  * **Inverse Mapping & Output Range Policy**: $[0.0, 1.0] \rightarrow [-1000.0, +1000.0]$ HU with `DOMAIN_BOUNDED_CLAMPING`. Linear residual addition without saturation layers can produce minor numerical overshoot in background air ($-1010$ HU) or dense cortical bone ($+1027$ HU). Normalized predictions are bounded to $[0.0, 1.0]$ matching the calibration domain; all raw excursions and pixel counts are explicitly accounted for in `output_range_accounting`.
  * **Loss Function**: Compound edge loss: Huber loss ($\delta=0.01$) + $0.01 \times$ Vector Gradient L1 loss.
* **Dedicated Research Entry Point & Upload Hardening**:
  * **Endpoint**: `POST /api/research/denoise_dicom` (Flask API).
  * **Supported Format**: 2D single-slice axial CT DICOM (`Modality == 'CT'`) with valid numeric `RescaleSlope` and `RescaleIntercept`.
  * **Payload Bound**: Strictly capped at 32 MB per request (`MAX_DICOM_UPLOAD_SIZE = 32 MB`). Payloads exceeding this limit receive HTTP 413.
  * **Unsupported / Rejected Formats**: Ordinary 8-bit PNG/JPEG/BMP images, non-CT modalities (MR, CR, XA), multiframe volumes (`NumberOfFrames > 1`), non-axial/3D arrays, and uncalibrated DICOM without rescale tags.
  * **Ephemeral In-Memory Storage**: Scans uploaded to the research route are processed strictly in-memory (or in transient unlinked buffers) and are never stored persistently on disk.
  * **Selected Metadata Redaction & Whitelisting**: All direct patient identifiers (`PatientName`, `PatientBirthDate`, `PatientID`), physician/institution fields, accession numbers, study/series/SOP UIDs, and private tags are excluded/redacted (`patient_id: "[REDACTED_FOR_PRIVACY]"`). Only non-identifying spatial geometry and numeric calibration parameters are emitted.
  * **Privacy Limitations & De-identification Distinction**: Formal DICOM de-identification (e.g. DICOM PS 3.15 Basic Application Level Confidentiality Profile) is NOT certified. Burned-in pixel annotations (if present in input scans) are rendered into 8-bit preview images without OCR or pixel scrubbing. Absolute claims of "zero PHI" or "HIPAA compliance" are avoided.
  * **Error Handling**: Controlled, sanitized diagnostic error messages without server path, file system, or stack trace leakage.

* **Dataset Splits & Provenance**:
  * **Dose Simulation Provenance**: The TCIA LDCT low-dose scans were reconstructed from projection sinogram data with simulated lower dose (e.g., realistic Poisson noise insertion into detector channels), not from an independent second physical CT scan of the patient.
  * **Development Training**: `data/manifests/tcia_residual_dev_split_manifest.json` (10 patients, 6,996 patches, 12 epochs).
  * **Development Validation**: `data/manifests/tcia_residual_dev_split_manifest.json` (2 patients, 1,296 patches).
  * **Locked Fresh Test Cohort**: `data/manifests/tcia_residual_v3_locked_test_manifest.json` (3 independent patients: `L179`, `L210`, `L057`; **441 paired CT slices**).
  * **Disjointness**: 100% patient-disjoint separation across all train, validation, and test cohorts ($\text{Intersection} = \emptyset$). Zero data leakage.
* **Verified Benchmark Metrics Across Domains (441 Paired Test Slices)**:
  * **Strictly Consistent Soft-Tissue Domain** (Window $[-160, 240]$ HU, Range $400.0$ HU):
    * ST-MAE: Raw $13.76$ HU $\rightarrow$ v3 **$8.89$ HU** (**$-35.39\%$** reduction).
    * ST-RMSE: Raw $17.60$ HU $\rightarrow$ v3 **$11.49$ HU** (**$-34.72\%$** reduction).
    * Consistent ST-PSNR: Raw $27.25$ dB $\rightarrow$ v3 **$30.89$ dB** (**$+3.64$ dB** gain).
    * Consistent ST-SSIM: Raw $0.8776 \rightarrow$ v3 **$0.9250$** (**$+0.0474$** gain).
    * ST Vector Gradient Error: Raw $93.37\ \text{HU/px} \rightarrow$ v3 **$74.39\ \text{HU/px}$** (**$-20.33\%$** reduction).
  * **Strictly Consistent Full-Body Domain** (Mask $\text{HU} > -500$, Range $2000.0$ HU):
    * Body MAE: Raw $13.82$ HU $\rightarrow$ v3 **$9.54$ HU** (**$-30.97\%$** reduction).
    * Body RMSE: Raw $17.73$ HU $\rightarrow$ v3 **$15.16$ HU** (**$-14.49\%$** reduction).
    * Body PSNR: Raw $41.16$ dB $\rightarrow$ v3 **$43.00$ dB** (**$+1.84$ dB** gain).
    * Full-Scan SSIM: Raw $0.9641 \rightarrow$ v3 **$0.9857$** (**$+0.0216$** gain).
    * Full-Body Vector Gradient Error: Raw $97.57\ \text{HU/px} \rightarrow$ v3 **$78.01\ \text{HU/px}$** (**$-20.05\%$** reduction).
  * **Mathematical Reconciliation of Previously Reported Mixed PSNR**:
    * Previously reported mixed PSNR ($44.87$ dB vs raw $41.22$ dB) combined full-body dynamic range ($2000.0$ HU) with soft-tissue RMSE in the denominator.
    * The exact shift is $20 \log_{10}(2000 / 400) = 20 \log_{10}(5) = \mathbf{+13.98\ \text{dB}}$ across all models.
* **Research Status & Outstanding External Validation Requirements**:
  * **Status**: **PRELIMINARY RESEARCH CANDIDATE**.
  * **Outstanding Requirements**: Multi-vendor scanner evaluation (GE, Siemens, Philips, Canon), multi-site clinical diversity, evaluation across diverse anatomical pathologies (e.g. lung nodule, brain hemorrhage, liver lesions), and formalized radiologist blind grading.
  * **Regulatory & Clinical Safety Notice**: FOR INVESTIGATIONAL / RESEARCH USE ONLY. Not certified as a medical device for clinical diagnosis, patient management, or clinical decision support under FDA/CE regulations. Never present denoised output as an approved replacement for original clinical CT scans.

---

## 3. Legacy Data Cleanup Manifest Summary
* **Action**: Executed safe removal of demonstrably invalid derived data (`data/separated_scans/brain/`).
* **Removed Files**: 952 files (30.14 MB) containing mislabeled Pelvis CT scans.
* **Preserved Assets**:
  * `data/DL_info.csv`: NIH DeepLesion official metadata (32,735 records).
  * `data/unified_dataset/clean/` & `data/unified_dataset/noisy/`: 2,069 authentic source CT scans.
  * `data/processed/deeplesion_localization/`: 2,069 validated authentic slices across patient-disjoint splits.
  * All root and output model checkpoints.
* **Documentation**: See `data/manifests/cleanup_manifest.json` (local working tree artifact).
