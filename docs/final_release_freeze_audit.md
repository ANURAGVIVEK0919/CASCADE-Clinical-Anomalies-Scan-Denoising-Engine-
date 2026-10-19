# CASCADE — Final Release Freeze, Acceptance Criteria & Privacy Consistency Audit

**Document ID:** `DOC-RELEASE-FREEZE-2026-10-20`
**Date:** 2026-10-20
**Project Path:** `D:\CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-`
**Branch:** `feat/model-rag-quality`
**Target Checkpoint:** `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`
**Expected SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
**Verified SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea` (Exact match)

---

## 1. Status Classifications

To ensure unambiguous regulatory, operational, and engineering clarity, this audit assigns three distinct status classifications:

| Category | Classification | Description |
| :--- | :--- | :--- |
| **Preliminary Research Status** | **`PRELIMINARY_RESEARCH_ACCEPTED`** | The candidate Residual U-Net v3 model successfully meets all prespecified engineering acceptance criteria on the locked 441-slice held-out research cohort (`L179`, `L210`, `L057`). |
| **Release Documentation Status** | **`RELEASE_DOCUMENTATION_PASS`** | All evaluation metrics, aggregation methodologies, privacy boundaries, and repository inventories are reconciled and verified with clean tests (89/89) and git diff whitespace checks. |
| **Clinical Validation Status** | **`CLINICAL_VALIDATION_NOT_ESTABLISHED`** | The system is an experimental research prototype. It has **NOT** undergone prospective multi-vendor clinical trials and is **NOT** FDA or CE cleared for medical diagnosis. |

---

## 2. Gate 1 — Acceptance Criteria Provenance Reconciliation

An audit of executable evaluation code, manifests, and historical audit reports was conducted to trace the origin of reported acceptance thresholds.

### 2.1 The Criteria Discrepancy
Two differing standards for Criterion 1 (Soft-Tissue MAE) appear in the repository history:
1. **Rule A (Prespecified):** `Soft-Tissue MAE <= 10.0 HU AND improvement >= 0.50 HU over raw low-dose CT`.
2. **Rule B (Retrospective Variant):** `Soft-Tissue MAE reduction >= 20% relative to raw low-dose CT`.

### 2.2 Provenance Investigation & Findings
- **Rule A Provenance:** Prespecified directly in executable code before the test cohort was evaluated. It is codified in:
  - `evaluate_residual_unet_native.py` (line 24): Written prior to native training and locked testing.
  - `evaluate_residual_unet_v3.py` (lines 14 & 225): Codified in the test script:
    ```python
    crit1_pass = (cand_st_mae <= 10.0) and (mae_diff <= -0.50)
    ```
  - `docs/unet_residual_v3_edge_audit.json` (line 106).
  - **Verdict:** `PRESPECIFIED_AND_VERIFIED`.
- **Rule B Provenance:** First appeared in `docs/unet_residual_v3_final_acceptance_audit.md` and `.json` in the audit summary tables. No prior commit, configuration file, or executable script specified a 20% relative reduction rule prior to observing the test results.
  - **Verdict:** `THRESHOLD_PROVENANCE_UNVERIFIED`.
- **Mathematical Impact:** On the locked 441-slice cohort, the raw low-dose ST-MAE was $13.76$ HU, and Residual U-Net v3 achieved **$8.89$ HU** (a reduction of **$4.87$ HU** or **$35.39\%$**). The candidate model mathematically satisfies both Rule A and Rule B. However, to maintain scientific rigor, Rule B is formally marked `THRESHOLD_PROVENANCE_UNVERIFIED` and Rule A is affirmed as the sole prespecified engineering acceptance criterion.

| Criterion | Prespecified Requirement | Provenance Status | Measured Candidate Value | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Criterion 1 (Prespecified)** | ST-MAE $\le 10.0$ HU and improvement $\ge 0.50$ HU | `PRESPECIFIED_AND_VERIFIED` | **8.89 HU** (4.87 HU reduction) | **PASS** |
| **Criterion 1 (Retrospective)** | ST-MAE reduction $\ge 20\%$ | `THRESHOLD_PROVENANCE_UNVERIFIED` | **35.39% reduction** | **MATHEMATICALLY MET** |
| **Criterion 2 (Structural)** | SSIM $\ge 0.9000$ and degradation $\ge -0.015$ | `PRESPECIFIED_AND_VERIFIED` | **0.9857** (+0.0216 vs raw) | **PASS** |
| **Criterion 3 (Edge Preservation)**| ST Vector Gradient Error $\le$ Raw Low-Dose CT | `PRESPECIFIED_AND_VERIFIED` | **74.39** vs 93.37 HU/px ($-20.33\%$) | **PASS** |

---

## 3. Gate 2 — PSNR and Metric Aggregation Analysis

### 3.1 Distinct PSNR Domains
PSNR calculations depend critically on the dynamic range $\text{MAX}_I$ and the anatomical mask applied. Two distinct variants exist in the CASCADE documentation and must never be conflated:

1. **Strictly Consistent Soft-Tissue PSNR:**
   - **Intensity Domain:** Abdominal Soft-Tissue Window $[-160, +240]$ HU.
   - **Peak Dynamic Range:** $\text{MAX}_I = 400.0$ HU ($240 - (-160)$ HU).
   - **Mask:** Ground-truth soft-tissue voxels $[-160, +240]$ HU within patient body contour ($\text{HU} > -500$).
   - **Formula:** $\text{PSNR}_{\text{ST}} = 20 \log_{10}(400.0 / \text{RMSE}_{\text{ST}})$.
   - **Values:** Raw Low-Dose CT = **27.25 dB**; Residual U-Net v3 = **30.89 dB** (+3.64 dB gain).
2. **Historical Mixed-Domain PSNR:**
   - **Intensity Domain:** RMSE evaluated on soft tissue, but scaled against the full clinical window $[-1000, +1000]$ HU.
   - **Peak Dynamic Range:** $\text{MAX}_I = 2000.0$ HU.
   - **Formula:** $\text{PSNR}_{\text{mixed}} = 20 \log_{10}(2000.0 / \text{RMSE}_{\text{ST}})$.
   - **Values:** Raw Low-Dose CT = **41.22 dB**; Residual U-Net v3 = **44.87 dB** (+3.65 dB gain).
   - **Mathematical Relationship:**
     $$\text{PSNR}_{\text{mixed}} = \text{PSNR}_{\text{ST}} + 20 \log_{10}\left(\frac{2000.0}{400.0}\right) = \text{PSNR}_{\text{ST}} + 20 \log_{10}(5) = \text{PSNR}_{\text{ST}} + 13.9794\ \text{dB}$$

### 3.2 Aggregation Procedures: Slice-Averaging vs. Pooled RMSE
- **Slice-Averaged PSNR:** In `evaluate_residual_unet_v3.py`, PSNR is computed individually for each slice $i$, and the arithmetic mean $\frac{1}{N}\sum \text{PSNR}_i$ is taken across the 441 slices, yielding **30.89 dB**.
- **Pooled RMSE PSNR:** If the pooled mean RMSE ($11.49$ HU) is used directly, $\text{PSNR} = 20\log_{10}(400.0 / 11.49) =$ **30.83 dB**.
- **Mathematical Cause of Difference:** The function $f(x) = 20 \log_{10}(\text{MAX} / x)$ is strictly convex for $x > 0$. By **Jensen’s inequality**:
  $$\mathbb{E}[f(x)] \ge f(\mathbb{E}[x])$$
  The mean of slice PSNRs will mathematically exceed the PSNR of the mean RMSE by $+0.06$ dB. Both conventions are mathematically valid but represent different operational aggregations.

### 3.3 Patient-Level Aggregation Weighting
The locked test cohort consists of:
- Patient `L179`: 143 slices ($32.43\%$)
- Patient `L210`: 149 slices ($33.79\%$)
- Patient `L057`: 149 slices ($33.79\%$)
- **Micro-average (equal weight per slice across 441 slices):** ST-MAE = **8.89 HU**, SSIM = **0.9857**.
- **Macro-average (equal 1/3 weight per patient):** ST-MAE = $\frac{8.77 + 9.83 + 8.07}{3} =$ **8.89 HU**, SSIM = $\frac{0.9883 + 0.9820 + 0.9868}{3} =$ **0.9857**.
Because the slice counts are virtually balanced across the three patients, micro-averaging and macro-averaging produce identical results at standard reporting precision.

---

## 4. Gate 3 — Output File Retention and Privacy Consistency

The privacy and retention policies of the research endpoint (`POST /api/research/denoise_dicom`) were audited against code paths in `research_inference.py` and `app.py`.

### 4.1 Distinct Technical Layers
1. **Source-File Immutability:** Input scans and reference datasets are accessed read-only. Source files are never modified, overwritten, or moved.
2. **In-Memory Input Processing:** Uploaded multipart byte streams are read directly into memory (`io.BytesIO`) using bounded streaming (max 32 MB). Scans are not stored in temporary files or persistent upload directories.
3. **Derived Output Storage Behavior:**
   - In `POST /api/research/denoise_dicom`, processing is **strictly ephemeral**.
   - Output arrays are converted to base64-encoded PNG display previews and JSON numerical statistics returned directly in the HTTP response.
   - **Correction of Prior Inaccuracy:** A prior audit draft mentioned that derived DICOM files were written to `data/output/research_dicom/`. Code inspection verified that **no such directory exists and no files are written to disk** during API execution. The API is entirely ephemeral.
4. **DICOM Metadata Redaction:** Whitelist filtering selectively retains only non-identifying physical CT calibration parameters (`RescaleSlope`, `RescaleIntercept`) and spatial geometry (`PixelSpacing`, `SliceThickness`, `SliceLocation`). Direct patient identifiers (`PatientName`, `PatientID`, `PatientBirthDate`), physician names, institution metadata, and private vendor tags are completely excluded from payloads and logs.
5. **Formal De-Identification Status:** The application **DOES NOT** certify compliance with DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile. Burned-in pixel annotations (e.g., patient name or study date burned directly into pixel intensities at the scanner console) are not scrubbed and will appear in 8-bit preview renderings if present in input scans. All unsupported absolute claims (such as "zero PHI", "fully de-identified", or "HIPAA compliant") are strictly disclaimed.

---

## 5. Gate 4 — Frozen Checkpoint Integrity & Repository Audit

### 5.1 Checkpoint Integrity Verification
- **Target File:** `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`
- **Expected SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
- **Computed SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
- **Integrity Status:** **VERIFIED & IMMUTABLE**.

### 5.2 Test Suite Execution
- **Command:** `& "venv\Scripts\python" -m unittest discover tests`
- **Result:** **89 tests passed in 58.126s (0 failures, 0 errors)**.
- **Coverage:** Full test coverage across security hardening, pipeline isolation, HU calibration, patch extraction, Hann blending, and model architectures.

### 5.3 Git Whitespace & Ignore Verification
- **Git Diff Check:** `git diff --check` exited with clean return code **0**.
- **.gitignore Verification:** Confirmed that `data/`, `artifacts/`, `*.keras`, `*.dcm`, and `*.dicom` are explicitly ignored in `.gitignore`.

### 5.4 Working Tree Inventory & Classification (`git status --short`)

The working tree was audited to distinguish required project code, documentation, test suites, and temporary scratch utilities:

1. **Modified Tracked Files:**
   - `.gitignore`: Updated ignore rules for large weights and datasets.
   - `app.py`: Dedicated `/api/research/denoise_dicom` route and telemetry.
   - `evaluate_data.py`: Pre-existing evaluation utility.
   - `medical_agent.py`: LangGraph multi-agent diagnostic logic.
   - `pinecone_rag_engine.py`: RAG assistant implementation.
   - `validate_models_on_dataset.py`: Benchmark validation utility.
2. **Required Project Source (New Modules):**
   - `hu_calibration.py`: Physical Hounsfield Unit scaling and mask routines.
   - `research_inference.py`: Hardened research inference pipeline.
   - `models/residual_unet.py`: Residual U-Net architecture, patch grid, and 2D Hann blending.
   - `dataset_taxonomy.py`: Dataset schema and taxonomy definitions.
   - `knowledge_base/`: Clinical guideline and diagnostic reference documents.
   - `rag/`: Offline document retrieval engine.
3. **Audit Evidence & Documentation:**
   - `docs/final_release_freeze_audit.md` & `.json` (This audit).
   - `docs/post_clamping_benchmark_reconciliation.md` & `.json`.
   - `docs/final_hu_output_range_audit.md` & `.json`.
   - `docs/final_research_demo_readiness.md` & `.json`.
   - `docs/research_api_security_audit.md` & `.json`.
   - `docs/unet_residual_v3_final_acceptance_audit.md` & `.json`.
   - `docs/unet_residual_v3_edge_audit.md` & `.json`.
   - `docs/unet_residual_v3_integration_report.md` & `.json`.
   - `docs/model_dataset_registry.md`.
4. **Test Suites & Validation Utilities:**
   - `tests/test_research_api_security.py` (16 security & isolation tests).
   - `tests/test_residual_unet_v3_integration.py` (10 pipeline integration tests).
   - `tests/test_residual_unet_v3.py` (7 architecture & gradient tests).
   - `tests/test_residual_unet_native.py` (5 patch & Hann window tests).
   - `tests/test_hu_calibrated_pipeline.py` (13 calibration & mask tests).
   - `tests/test_model_rag_quality.py` (12 RAG & agent tests).
   - `evaluate_residual_unet_v3.py`: Executable benchmark evaluation script.
   - `evaluate_residual_unet_native.py`: Native patch evaluation script.
5. **Temporary Scratch Utilities for Owner Review (`scratch/`):**
   - `scratch/reconcile_clamped_benchmark.py`: Evaluator for clamped vs unclamped metrics.
   - `scratch/clamped_reconciliation_result.json`: Detailed 441-slice reconciliation cache.
   - `scratch/test_real_dicom_smoke.py`: Local smoke test utility.
   - `scratch/download_*.py`, `scratch/build_*.py`, `scratch/inspect_*.py`: Data download and inspect scripts.
   - *(Note: Scratch files are retained for developer inspection; no files were deleted).*

---

## 6. Gate 5 — Final Freeze Declaration & Regulatory Disclaimers

### 6.1 Freeze Declaration
The Edge-Preserving Residual U-Net v3 model checkpoint (`candidate_unet_residual_edge_v3.keras`, SHA-256 `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`) is formally declared **FROZEN** as a **Preliminary Research Candidate**.
- **Model Retraining:** Prohibited.
- **Weight Fine-Tuning:** Prohibited.
- **Architecture / Hyperparameter Changes:** Prohibited.
- **Deployment Status:** Local research prototype only; no external deployment or distribution.

### 6.2 Mandatory Clinical & Regulatory Disclaimers

> [!WARNING]
> **INVESTIGATIONAL RESEARCH PROTOTYPE ONLY**
> This software system, neural network model, and accompanying scripts are strictly investigational tools designed for retrospective research benchmarking.
>
> 1. **No Regulatory Clearance:** This system is **NOT** cleared, approved, or certified by the U.S. Food and Drug Administration (FDA), European Medicines Agency (CE mark), or any national health authority.
> 2. **Not for Diagnostic Use:** Denoised outputs and AI diagnostic findings must **NEVER** be used as a substitute for primary radiological review, clinical diagnosis, patient triage, or medical treatment planning.
> 3. **Cohort Generalization Scope:** Validation is strictly limited to 441 retrospective paired slices from 3 patients in the TCIA LDCT simulated low-dose challenge (Siemens SOMATOM Definition AS+ reconstruction). Generalization across physical low-dose repeated scans, diverse patient body habitus, pediatric cohorts, or multi-vendor scanners (GE, Philips, Canon) is unproven and must not be inferred.
> 4. **Ancillary System Blockers:** Cranial CT evaluation remains offline due to the absence of authentic matched head CT pairs.
