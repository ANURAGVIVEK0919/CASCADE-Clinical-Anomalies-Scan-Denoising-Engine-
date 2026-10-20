# CASCADE — Final Benchmark Metric Integrity & Reproducibility Audit

## Executive Summary

This audit establishes mathematical rigor, metric domain truthfulness, and operational safety for the canonical Residual U-Net v3 training and evaluation tooling (`train_residual_unet_v3.py` and `evaluate_residual_unet_v3.py`).

All non-negotiable safety rules were respected:
- **Zero Model Training / Checkpoint Modification:** No training was launched, and the frozen candidate checkpoint hash (`bc396511...`) remains identical and intact.
- **No Evaluation on Locked Test Cohort:** All metric evaluations and unit tests were conducted strictly using synthetic numpy arrays and isolated temporary directory fixtures.
- **Metric Honesty:** The distinction between consistent soft-tissue PSNR (400 HU range) and historical mixed-domain PSNR (2000 HU range) is made mathematically explicit, accounting for the +13.98 dB constant offset.
- **Staging / Remote Invariance:** Zero files were staged, committed, or pushed.

---

## 1. PSNR Calculation & Domain Audit

### Formula & Domain Traceability
In `evaluate_residual_unet_v3.py`, error computation and peak signal metrics are calculated via `compute_slice_metrics()`:

| Component | Physical Soft-Tissue Domain | Historical Mixed Domain | Whole-Body Domain |
| :--- | :--- | :--- | :--- |
| **Field Name** | `consistent_st_psnr_db` | `historical_mixed_psnr_db` (`psnr_db`) | `global_body_psnr_db` |
| **Mask Applied** | Soft-Tissue Mask (`[-160, +240]` HU) | Soft-Tissue Mask (`[-160, +240]` HU) | Patient Body Contour Mask |
| **Error Domain** | Soft-Tissue RMSE ($\text{RMSE}_{\text{ST}}$) | Soft-Tissue RMSE ($\text{RMSE}_{\text{ST}}$) | Body Contour RMSE ($\text{RMSE}_{\text{body}}$) |
| **Numerator ($\text{MAX}_I$)** | **400.0 HU** ($240 - (-160)$) | **2000.0 HU** ($1000 - (-1000)$) | **2000.0 HU** ($1000 - (-1000)$) |
| **Formula** | $20 \log_{10}\left(\frac{400.0}{\text{RMSE}_{\text{ST}} + 10^{-8}}\right)$ | $20 \log_{10}\left(\frac{2000.0}{\text{RMSE}_{\text{ST}} + 10^{-8}}\right)$ | $20 \log_{10}\left(\frac{2000.0}{\text{RMSE}_{\text{body}} + 10^{-8}}\right)$ |
| **Aggregation Method** | Per-slice evaluation, then slice-averaged mean across cohort | Per-slice evaluation, then slice-averaged mean across cohort | Per-slice evaluation, then slice-averaged mean across cohort |

### Mathematical Offset Derivation
The difference between the historical mixed-domain PSNR and the consistent soft-tissue PSNR is a fixed mathematical constant:

$$\Delta\text{PSNR} = 20 \log_{10}(2000.0) - 20 \log_{10}(400.0) = 20 \log_{10}\left(\frac{2000.0}{400.0}\right) = 20 \log_{10}(5) \approx 13.9794\text{ dB}$$

$$\text{historical\_mixed\_psnr\_db} = \text{consistent\_st\_psnr\_db} + 13.98\text{ dB}$$

### Metric Semantics & Naming Policy
1. `consistent_st_psnr_db`: The true, physical soft-tissue PSNR where both dynamic range numerator and error denominator correspond to the 400 HU soft-tissue window.
2. `historical_mixed_psnr_db`: Retained strictly for backwards-compatible comparison against historical run records and acceptance reports, explicitly documented as combining whole-body dynamic range with soft-tissue error.
3. `psnr_db`: Maintained as an alias to `historical_mixed_psnr_db` to ensure automated legacy report parsers remain functional.

### Provenance Distinction: Verified vs Historical Records
- **Directly Verified (Current Environment):**
  - Mathematical relationship: $\text{historical\_mixed\_psnr\_db} - \text{consistent\_st\_psnr\_db} = 20 \log_{10}(2000 / 400) = 13.9794\text{ dB}$ (verified on synthetic arrays to $< 10^{-4}$ precision).
  - Shape validation, non-finite array rejection (`ValueError`), and empty-mask `None` (`null`) semantics in `compute_slice_metrics`.
  - Aggregation integrity: Empty-mask slices reporting `None` are excluded from cohort averages in `aggregate_cohort_metrics` without biasing valid slice means, valid counts are tracked, and all-empty cohorts return `INSUFFICIENT_DATA`.
  - Identity initialization and dry-run isolation logic in `train_residual_unet_v3.py`.
  - Checkpoint integrity: SHA-256 hash `bc396511...` matched bit-for-bit.
- **Historical Benchmark Record (Recorded Prior Evaluation):**
  - The historical figures ($30.89$ dB consistent ST-PSNR / $44.87$ dB mixed-domain PSNR, $+3.64$ dB gain across 441 paired TCIA slices) originate from prior runs recorded in `docs/unet_residual_v3_final_acceptance_audit.json` and `docs/final_release_freeze_audit.json`.
  - In compliance with the mandatory safety rules, the 441-slice locked patient cohort was **NOT re-evaluated** during this audit. These historical numbers are documented as an archival benchmark record (`HISTORICAL_RECORD`), not re-generated.

### Empty Evaluation Mask Policy & Aggregation Semantics
- **Per-Slice Semantics:** When a CT slice contains zero valid pixels for a specific evaluation mask (e.g. no soft-tissue voxels in `[-160, +240]` HU or no body contour), the metric is **unavailable** and returned as `None` (`null` in serialized JSON).
- **No Fabricated Zeros:** Slices with empty masks NEVER report MAE or RMSE as `0.0`, nor a fabricated ceiling PSNR (~212 dB).
- **Cohort Mean Calculation:** In `aggregate_cohort_metrics`, averages are computed strictly over slices with valid, non-null measurements. Valid slice counts (`valid_slice_counts`) are recorded per metric.
- **Insufficient Data Guard:** If all slices in an evaluation cohort have empty masks for a required metric, the aggregate returns `None` and the acceptance audit flags `overall_status: INSUFFICIENT_DATA` rather than returning a misleading numeric pass/fail.

---

## 2. Structural & Edge Metric Implementation Audit

### SSIM (`ssim`)
- **Inputs:** Native $512 \times 512$ full slices linearly mapped to $[0, 1]$ via $\frac{\text{clip}(\text{HU}, -1000, 1000) + 1000}{2000}$.
- **Masking:** Unmasked (evaluated across the entire slice matrix).
- **Dynamic Range:** `max_val = 1.0`.
- **Implementation:** `tf.image.ssim(t_pred, t_ref, max_val=1.0)`.
- **Aggregation:** Arithmetic mean of slice SSIM values across the cohort.

### Vector Gradient Discrepancy (`vector_gradient_error`)
- **Inputs:** Unclipped Hounsfield Unit slices.
- **Gradient Operator:** 2D Sobel $3 \times 3$ derivative filters ($\text{CV\_32F}$) computing $g_x = \frac{\partial I}{\partial x}$ and $g_y = \frac{\partial I}{\partial y}$.
- **Norm:** Euclidean $L_2$ vector discrepancy per pixel:
  $$\text{error}_i = \sqrt{(g_{x, \text{pred}} - g_{x, \text{ref}})^2 + (g_{y, \text{pred}} - g_{y, \text{ref}})^2}$$
- **Masking:** Evaluated over high-gradient boundary edge pixels (top 25th percentile of Sobel magnitude inside the soft-tissue mask).
- **Units:** $\text{HU/pixel}$.
- **Aggregation:** Arithmetic mean across all masked boundary pixels, averaged across valid slices.

---

## 3. Training Script Safety & Guard Verification

Inspection and testing of `train_residual_unet_v3.py` confirmed:
1. **Safe Dry-Run by Default:** Invoking the script without `--execute` runs configuration validation only. It audits patient-disjoint separation across dev, v2, and v3 manifests without allocating GPU tensors, building training datasets, or writing checkpoints.
2. **Explicit Opt-in Execution:** Training is locked behind `--execute`. Accidental script execution or import cannot trigger training epochs.
3. **Anti-Overwrite Protection:** If `--checkpoint-path` already exists on disk, training is blocked unless `--overwrite` is explicitly passed.
4. **Cohort Isolation Audit:** Manifests are read strictly to assert cohort disjointness (`train.isdisjoint(val)` and `train.isdisjoint(locked_test)`). The locked test cohorts are never accessed for training or hyperparameter selection.

---

## 4. Verification Suite Results

### A. Focused Tooling Tests (`tests/test_research_reproducibility_tooling.py`)
```powershell
& "venv\Scripts\python" -m unittest -v tests/test_research_reproducibility_tooling.py
```
- **Total Tests:** **16/16 passed (OK)** in 0.218s.
- **Coverage:**
  - Dry-run mode without training execution or checkpoint generation.
  - Patient leakage assertion on overlapping cohort manifests.
  - Checkpoint anti-overwrite guard (`CHECKPOINT_EXISTS_ABORTED`).
  - Missing manifest and missing DICOM handling (`DATA_UNAVAILABLE`).
  - Safe report overwrite prevention (`overwrite=False` vs `overwrite=True`).
  - Numerical validation of PSNR formulas and the exact $+13.98\text{ dB}$ offset on controlled synthetic arrays.
  - Semantic integrity of report documentation fields.
  - Empty soft-tissue mask returning `None` without fabricating `0.0` or PSNR.
  - Empty body mask returning `None` for global body metrics.
  - Mixed valid and empty-mask slices confirming valid means are not diluted.
  - All-empty cohorts returning `None` aggregate and `INSUFFICIENT_DATA`.
  - JSON serialization and deserialization of `null` metrics without corruption.
  - Slice/mask shape mismatch `ValueError`.
  - Non-finite NaN/Inf array `ValueError`.

### B. Full Test Discovery Suite
```powershell
& "venv\Scripts\python" -m unittest discover tests
```
- **Total Tests:** **117/117 passed (OK)** in 71.561s.
- **Test Modules:** All existing HU calibration, research API security, U-Net native, U-Net v3 integration, dataset validation, and reproducibility tests pass with zero regressions.

### C. Whitespace & Formatting Audit
1. **Tracked Files (`git diff --check`):**
   - Result: Clean (Exit code 0).
2. **Untracked Candidate Tooling Files:**
   - Evaluated: `train_residual_unet_v3.py`, `evaluate_residual_unet_v3.py`, `tests/test_research_reproducibility_tooling.py`, `docs/benchmark_metric_integrity_audit.md`.
   - Result: 100% clean of trailing whitespace and tab characters.

### D. Frozen Model Checkpoint SHA-256 Invariance
Path: `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`
- Target: `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
- Calculated: `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
- **Verification:** `MATCH: True` (100% frozen and intact).

---

## 5. Working Tree & Git Status

- **Active Branch:** `feat/research-reproducibility-tooling`
- **Base HEAD:** `dd09b63`
- **Staging Status:** Clean (`git diff --cached --stat` empty; 0 files staged).
- **Candidate Files for Future Changeset:**
  1. `train_residual_unet_v3.py` (untracked)
  2. `evaluate_residual_unet_v3.py` (untracked)
  3. `tests/test_research_reproducibility_tooling.py` (untracked)
  4. `docs/benchmark_metric_integrity_audit.md` (untracked)
- **Unresolved Blockers:** None.

---

## 6. Research Status & Non-Clinical Disclaimer

This audit document, associated scripts, and evaluation metrics are produced strictly for pre-clinical academic and engineering reproducibility research.
- **NOT a Medical Device:** The CASCADE denoising models and scripts have not been evaluated, cleared, or approved by the FDA, CE, or any regulatory body.
- **NO Clinical Diagnostic Use:** Output scans, Hounsfield Unit adjustments, and diagnostic overlays must not be used for primary diagnostic interpretation, treatment planning, or clinical triage.
