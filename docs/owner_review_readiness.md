# CASCADE — Final Owner-Review Readiness & Acceptance Provenance Audit

**Document ID:** `DOC-OWNER-REVIEW-2026-10-20`
**Date:** 2026-10-20
**Project Path:** `D:\CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-`
**Branch:** `feat/model-rag-quality`
**Current HEAD:** `0ede53e`
**Target Frozen Checkpoint:** `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`
**Expected SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
**Verified SHA-256:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea` (Exact match)
**Audit Purpose:** Provide an unvarnished, evidence-backed evaluation of acceptance criteria temporal provenance, metric transparency, large-diff architectural risks, and staging classifications for repository owner review.

---

## 1. Gate 1 — Temporal Provenance Audit of Acceptance Criteria

A rigorous separation was conducted between **code implementation** and **independent historical timing evidence** for the locked 441-slice held-out test cohort (`L179`, `L210`, `L057`):

### 1.1 Temporal Context & Working-Tree Evidence
- All development and evaluation of Residual U-Net v3 resides in the local uncommitted working tree branching off commit `0ede53e` (dated Oct 19, 2026, 16:03:20 +0530).
- File system creation timestamps show:
  * `evaluate_residual_unet_native.py` (ctime `2026-10-19 23:28:41`) codified MAE gain $\ge 0.50$ HU and SSIM degradation $\le 0.015$.
  * `tcia_residual_v3_locked_test_manifest.json` (ctime `2026-10-20 00:17:47`).
  * `candidate_unet_residual_edge_v3.keras` (ctime `2026-10-20 00:26:56`, training finished mtime `01:10:03`).
  * `evaluate_residual_unet_v3.py` (ctime `2026-10-20 00:30:08`) codified MAE $\le 10.0$ HU, MAE gain $\ge 0.50$ HU, SSIM $\ge 0.9000$, SSIM degradation $\le 0.015$, and Vector Grad Error $\le$ raw baseline.
  * Test execution completed around `01:15` - `01:30`, producing `docs/unet_residual_v3_edge_audit.md` (ctime `01:31:32`).
- **Core Forensic Finding:** While the thresholds were encoded in `evaluate_residual_unet_v3.py` before execution of the test script, **both the test script and the evaluation results were generated within the same uncommitted working tree session without an independent third-party timestamp or an immutable Git commit prior to running the test**. Therefore, in accordance with scientific audit standards, their temporal prespecification cannot be independently established.

### 1.2 Criteria Provenance Classification Table

| Criterion | Rule Definition | In Current Executable Code? | Independent Prior Record? | Working-Tree Evidence | Temporal Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **ST-MAE $\le 10.0$ HU** | Mean soft-tissue intensity error $\le 10.0$ HU | **YES** (`evaluate_residual_unet_v3.py:L226`) | **NO** | Codified in test runner script | `NOT_INDEPENDENTLY_VERIFIED` |
| **ST-MAE Gain $\ge 0.50$ HU** | MAE improvement over raw $\ge 0.50$ HU ($\Delta \le -0.50$) | **YES** (`evaluate_residual_unet_v3.py:L226`, `native:L290`) | **NO** | Present across v2 and v3 runners | `NOT_INDEPENDENTLY_VERIFIED` |
| **Full SSIM $\ge 0.9000$** | Structural similarity on full scan $\ge 0.9000$ | **YES** (`evaluate_residual_unet_v3.py:L227`) | **NO** | Codified in test runner script | `NOT_INDEPENDENTLY_VERIFIED` |
| **SSIM Degradation $\le 0.015$** | Allowable structural loss relative to raw scan $\le 0.015$ ($\Delta \ge -0.015$) | **YES** (`evaluate_residual_unet_v3.py:L227`, `native:L291`) | **NO** | Present across v2 and v3 runners | `NOT_INDEPENDENTLY_VERIFIED` |
| **Vector Grad Error $\le$ Raw** | Vector directional gradient error does not exceed raw scan | **YES** (`evaluate_residual_unet_v3.py:L228`) | **NO** | Codified in test runner script | `NOT_INDEPENDENTLY_VERIFIED` |
| **Historical PSNR $\ge 42$ dB** | Mixed PSNR $\ge 42.0$ dB | **NO** | **NO** | Informal retrospective observation (v3 reached 44.87 dB vs raw 41.22 dB); never codified as a threshold | `UNVERIFIED_AS_CRITERION` |
| **Historical SSIM $\ge 0.95$** | SSIM $\ge 0.95$ | **NO** | **NO** | Informal retrospective observation (patient `L210` raw was 0.9564); never codified as a threshold | `UNVERIFIED_AS_CRITERION` |
| **ST-MAE Reduction $\ge 20\%$** | Relative MAE reduction $\ge 20.0\%$ | **NO** | **NO** | First appeared post-hoc in `docs/unet_residual_v3_final_acceptance_audit.md` | `RETROSPECTIVE_POST_HOC_VARIANT` |

---

## 2. Gate 2 — Confirmed Metric Transparency & Limitations

Authoritative metrics evaluated on the locked 441-slice held-out cohort (`L179`, `L210`, `L057`):

| Evaluation Metric | Raw Low-Dose CT | Residual U-Net v3 | Net Improvement | Measurement Conditions & Scientific Limitations |
| :--- | :---: | :---: | :---: | :--- |
| **Soft-Tissue MAE (HU)** | 13.76 | **8.89** | **-4.87 HU (-35.39%)** | Abdominal soft-tissue $[-160.0, +240.0]$ HU inside patient body contour ($\text{HU} > -500$). |
| **Consistent Soft-Tissue PSNR (dB)** | 27.25 | **30.89** | **+3.64 dB** | Dynamic range $\text{MAX}_I = 400.0$ HU across $[-160.0, +240.0]$ HU window. |
| **Historical Mixed-Domain PSNR (dB)** | 41.22 | **44.87** | **+3.65 dB** | Full range $\text{MAX}_I = 2000.0$ HU in numerator with soft-tissue RMSE in denominator. Offset by $+13.98$ dB ($20\log_{10}(5)$); **not interchangeable** with soft-tissue PSNR. |
| **Consistent Soft-Tissue SSIM** | 0.8776 | **0.9250** | **+0.0474** | Soft-tissue window $[-160.0, +240.0]$ HU normalized over 400.0 HU width. **Primary recommended SSIM metric**. |
| **Reported Full-Scan SSIM** | 0.9641 | **0.9857** | **+0.0216** | Evaluated on full $512 \times 512$ canvas across $[-1000, +1000]$ HU. **Limitation:** Flat zero-valued ambient air outside the body inflates baseline scores; preserved in historical audit records but excluded from primary quality claims. |
| **Soft-Tissue Vector Gradient Error** | 93.37 | **74.39** | **-18.98 HU/px (-20.33%)** | Directional Sobel vector difference $\|\nabla I_{\text{pred}} - \nabla I_{\text{ref}}\|_2$. Units: **$\text{HU}/\text{pixel}$**. |

---

## 3. Gate 3 — Inspection of Large Existing Diffs Against HEAD

A line-by-line inspection of modified tracked files was conducted against `HEAD` (`0ede53e`):

### 3.1 `pinecone_rag_engine.py` (445 lines changed: +84, -361)
- **What Was Changed:**
  - Removed the hardcoded in-memory synthetic dataset `DENSE_CLINICAL_DATASET` (300+ lines).
  - Deleted `DenseEmbeddingGenerator` (hash-frequency projection).
  - Removed artificial score boosts (`+0.45` organ bonus, `+0.35` severity bonus) and synthetic fallback score `0.95`.
  - Connected `PineconeClinicalRAG` to `rag.retriever.ClinicalRAGRetriever` and genuine clinical guidelines.
- **Contract & Compatibility Changes:**
  - `similarity_score`: Previously returned a synthetic float (defaulting to `0.95`). Now returns `None` when offline or unconfigured.
  - `retrieval_mode`: Previously returned `'local_fallback'`. Now returns `'unconfigured'` or `'unavailable'`.
  - `matches_count`: Returns `0` when offline instead of synthetic matches.
- **Compatibility Audit:**
  - `medical_agent.py` was inspected and verified: lines 192–195 explicitly guard `if sim_score is not None: ... else: sim_score_str = "Not Available..."`. No crash occurs.
- **Owner Review Decision:** **`NEEDS_OWNER_REVIEW`** (Requires owner sign-off on the intentional transition from synthetic mock responses to authentic offline disclosures).

### 3.2 `validate_models_on_dataset.py` (253 lines changed: +127, -126)
- **What Was Changed:**
  - Replaced hardcoded machine paths (`data/separated_scans`) with CLI options (`--data-dir`, `--sample-size`, `--output-path`) and environment fallback `CASCADE_DATA_DIR`.
  - Renamed severity distribution keys from `{"Normal", "Mild Anomaly", "High Risk"}` to `{"Candidate ROI Detected", "No Candidate ROI Flagged"}`.
  - Restricted PSNR/SSIM calculation strictly to files where ground truth exists in `clean/`.
- **Compatibility Audit:**
  - No other python module imports `validate_models_on_dataset.py`. Standalone execution verified.
- **Owner Review Decision:** **`NEEDS_OWNER_REVIEW`** (Requires owner approval of the candidate ROI terminology).

### 3.3 `app.py` (182 lines changed: +181, -1)
- **What Was Changed:**
  - Added dedicated endpoint `POST /api/research/denoise_dicom` (32 MB upload limit, in-memory processing, selected metadata whitelist redaction, zero disk save).
  - Added `denoiser_engine` field to process responses (`"CASCADE U-Net Neural Denoiser"` vs. classical baseline).
  - Preserved all existing legacy routes (`/`, `/process`, `/api/s3/upload`) with zero breaking changes.
- **Owner Review Decision:** **`RECOMMEND_STAGE`**.

### 3.4 `evaluate_data.py` (180 lines changed: +108, -72)
- **What Was Changed:**
  - Made paths configurable via CLI args and environment variable `CASCADE_DATA_DIR`.
  - Added seeded random sampling (`random.seed(42)`) to replace unseeded `random.shuffle`.
- **Owner Review Decision:** **`NEEDS_OWNER_REVIEW`**.

### 3.5 `medical_agent.py` (63 lines changed: +39, -24)
- **What Was Changed:**
  - Removed synthetic default `ssim_delta = 0.267` -> now strictly `None` when clean reference is absent.
  - Handled `None` similarity scores safely without crashes.
  - Updated LLM prompt to explicitly forbid hallucinating coordinates or metrics.
- **Owner Review Decision:** **`RECOMMEND_STAGE`**.

### 3.6 `README.md` (151 lines changed: +92, -59)
- **What Was Changed:**
  - Full alignment with preliminary research candidate state, repository-relative links, metrics, privacy disclaimers, temporal provenance disclosure, and 89-test verification suite.
- **Owner Review Decision:** **`RECOMMEND_STAGE`**.

### 3.7 `.gitignore` (8 lines added)
- **What Was Changed:**
  - Added `scratch/` ignore rule. Preserves existing ignore rules for `data/`, `artifacts/`, `*.keras`, `*.dcm`.
- **Owner Review Decision:** **`RECOMMEND_STAGE`**.

---

## 4. Gate 4 — Audit of the Proposed Staging Inventory

Every file in the proposed 5 changesets is classified for repository owner review:

### Changeset 1: Core Physical HU Pipeline & Research DICOM Route
- `hu_calibration.py`: **`RECOMMEND_STAGE`** (Physical HU normalization/inversion)
- `models/residual_unet.py`: **`RECOMMEND_STAGE`** (Residual U-Net model definition)
- `research_inference.py`: **`RECOMMEND_STAGE`** (In-memory DICOM inference runner)
- `app.py`: **`RECOMMEND_STAGE`** (Dedicated research route)
- `.gitignore`: **`RECOMMEND_STAGE`** (`scratch/` protection)

### Changeset 2: Clinical Guidelines Knowledge Base & Local Offline RAG
- `knowledge_base/clinical_sources/*.md` (3 files): **`RECOMMEND_STAGE`** (ACR, Fleischner, LI-RADS summaries)
- `dataset_taxonomy.py`: **`RECOMMEND_STAGE`** (DeepLesion schema validator)
- `medical_agent.py`: **`RECOMMEND_STAGE`** (Safety guards, HITL sign-off)
- `pinecone_rag_engine.py`: **`NEEDS_OWNER_REVIEW`** (Synthetic fallback removal)
- `rag/` (8 modules): **`RECOMMEND_STAGE`** (Clean embedding and retrieval pipeline)

### Changeset 3: Comprehensive Offline Regression & Security Test Suite
- `tests/test_research_api_security.py`: **`RECOMMEND_STAGE`** (16 tests)
- `tests/test_residual_unet_v3_integration.py`: **`RECOMMEND_STAGE`** (10 tests)
- `tests/test_residual_unet_v3.py`: **`RECOMMEND_STAGE`** (11 tests)
- `tests/test_residual_unet_native.py`: **`RECOMMEND_STAGE`** (10 tests)
- `tests/test_hu_calibrated_pipeline.py`: **`RECOMMEND_STAGE`** (7 tests)
- `tests/test_model_rag_quality.py`: **`RECOMMEND_STAGE`** (22 tests)

### Changeset 4: Portable Evaluation, Dataset Integrity & Benchmark Scripts
- `evaluate_residual_unet_v3.py`: **`RECOMMEND_STAGE`** (Locked test evaluator)
- `evaluate_residual_unet_native.py`: **`RECOMMEND_STAGE`**
- `evaluate_hu_calibrated_unet.py`: **`RECOMMEND_STAGE`**
- `evaluate_authentic_suite.py`: **`RECOMMEND_STAGE`**
- `validate_dataset_integrity.py`: **`RECOMMEND_STAGE`**
- `evaluate_data.py`: **`NEEDS_OWNER_REVIEW`**
- `validate_models_on_dataset.py`: **`NEEDS_OWNER_REVIEW`**
- Historical scripts (`execute_safe_cleanup.py`, `prepare_authentic_dataset.py`, `train_*.py`, `evaluate_models.py`, `evaluate_residual_dev.py`): **`NEEDS_OWNER_REVIEW`** (Should only be executed deliberately; verify owner wishes to version historical training harnesses).

### Changeset 5: Model Registry & Audit Documentation
- `README.md`: **`RECOMMEND_STAGE`**
- `docs/model_dataset_registry.md`: **`RECOMMEND_STAGE`**
- `docs/dataset_sources.md`: **`RECOMMEND_STAGE`**
- Audit documents & JSON companions (13 pairs): **`RECOMMEND_STAGE`**
- `docs/owner_review_readiness.md` & `.json`: **`RECOMMEND_STAGE`**

### Files Protected & Excluded (Not for Staging)
- `scratch/*`: **`KEEP_IGNORED`** (Ignored via `.gitignore`, intact on disk)
- `data/*`: **`KEEP_IGNORED`** (Medical CT scans and manifests)
- `artifacts/*`: **`KEEP_IGNORED`** (Candidate model weights, including `candidate_unet_residual_edge_v3.keras`)
- `*.keras`: **`KEEP_IGNORED`** (Root model weights)
- `*.dcm`, `*.dicom`: **`KEEP_IGNORED`**
- `.env`: **`KEEP_IGNORED`** (Real secrets excluded; `.env.example` tracked)

---

## 5. Gate 5 — Final Non-Destructive Verification Summary

1. **Test Suite:** `& "venv\Scripts\python" -m unittest discover tests`
   - **Result:** **PASS (89/89 tests passed in 49.431s, 0 failures, 0 errors, 100% offline)**.
2. **Whitespace Check:** `git diff --check`
   - **Result:** Exit code **0** (clean, zero trailing whitespace or merge conflict markers).
3. **Checkpoint Integrity:** SHA-256 recalculation on `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`
   - **Result:** `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea` (**EXACT MATCH**).
4. **Git Index Cleanliness:** `git diff --cached --stat`
   - **Result:** **0 files staged**.
5. **Git Status:** `git status --short` confirms 7 tracked modified files and uncommitted untracked candidate files. No branch resets, pushes, or commits occurred.

---

## 6. Actionable Recommendations for Repository Owner

1. **Staging Execution Decision:** When ready, stage **Changesets 1, 2 (except review items), 3, and 5** (`RECOMMEND_STAGE` files).
2. **Review Items:** Inspect `pinecone_rag_engine.py` (synthetic fallback removal) and `validate_models_on_dataset.py` (candidate ROI terminology) prior to staging Changeset 2 and 4.
3. **Historical Training Scripts:** Decide whether historical training scripts (`train_residual_unet_v3.py`, etc.) should be committed for reproducibility or archived separately.
4. **Permanent Clinical Limitation:** Preserve the non-diagnostic research disclaimer in all future releases.
