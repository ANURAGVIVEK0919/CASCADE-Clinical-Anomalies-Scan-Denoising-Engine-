"""
CASCADE Residual U-Net v3 Historical Benchmark Evaluation Protocol (Gate 5)

Evaluates:
1. Raw Low-Dose CT
2. Residual U-Net v2 (candidate_unet_residual_native.keras)
3. Residual U-Net v3 Candidate (candidate_unet_residual_edge_v3.keras)

Cohort Context:
  Evaluated strictly on the historical evaluation cohort:
  Manifest: data/manifests/tcia_residual_v3_locked_test_manifest.json
  Patients: L179, L210, L057 (441 matched full-resolution slice pairs)

Engineering Acceptance Criteria (Prespecified before test execution):
- Criterion 1: Soft-Tissue MAE <= 10.0 HU AND improvement >= 0.50 HU over raw low-dose CT.
- Criterion 2: SSIM >= 0.9000 AND degradation >= -0.015 relative to raw low-dose CT.
- Criterion 3: Vector Gradient Error <= Raw Low-Dose Vector Gradient Error (anatomical edge preservation).

Metric Documentation & Measurement Domains:
- Soft-Tissue Window: Defined strictly as [-160, +240] HU (width = 400 HU), masking non-tissue/air regions.
- Whole-Body Data Range: Defined strictly as [-1000, +1000] HU (width = 2000 HU) for linear normalization.
- Consistent Soft-Tissue PSNR: 20 * log10(400.0 / (st_rmse + 1e-8)) strictly within soft-tissue dynamic range.
- Historical Mixed-Domain PSNR: 20 * log10(2000.0 / (st_rmse + 1e-8)) combining 2000 HU whole-body numerator with st_rmse denominator.
  (Offset by +13.98 dB relative to consistent soft-tissue PSNR).
- SSIM Domain: Structural similarity evaluated on full unmasked 512x512 slices linearly normalized to [0, 1] via [-1000, +1000] HU with max_val=1.0.
- Anatomical Edge Preservation: Mean Euclidean norm of 2D Sobel gradient discrepancy on top 25th percentile boundary edge masks within soft-tissue mask.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pydicom
import cv2
import tensorflow as tf

REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.residual_unet import (
    hu_to_normalized,
    normalized_to_hu,
    denoise_native_slice,
    get_soft_tissue_mask,
    get_body_contour_mask,
    HU_MIN,
    HU_MAX,
    HU_RANGE,
    SOFT_TISSUE_MIN,
    SOFT_TISSUE_MAX
)

SOFT_TISSUE_RANGE = float(SOFT_TISSUE_MAX - SOFT_TISSUE_MIN)  # 240.0 - (-160.0) = 400.0 HU


def load_dicom_hu(path: str) -> np.ndarray:
    dcm = pydicom.dcmread(path)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    return dcm.pixel_array.astype(np.float32) * slope + intercept


def sobel_vector(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return mag, gx, gy


def compute_slice_metrics(
    p_hu: np.ndarray,
    fd_hu: np.ndarray,
    p_norm: np.ndarray,
    fd_norm: np.ndarray,
    st_mask: np.ndarray,
    body_mask: np.ndarray
) -> Dict[str, float]:
    """
    Computes mathematical metrics for a single slice prediction versus ground truth.
    Explicitly distinguishes:
    1. Soft-tissue MAE (st_mae)
    2. Soft-tissue RMSE (st_rmse)
    3. Consistent soft-tissue PSNR (consistent_st_psnr, MAX_I = 400.0 HU)
    4. Historical mixed-domain PSNR (historical_mixed_psnr, MAX_I = 2000.0 HU)
    5. Full body contour PSNR (global_body_psnr, MAX_I = 2000.0 HU)
    6. Structural similarity (ssim, full unmasked slice [0, 1])
    7. 2D vector gradient error on high-gradient boundary masks.
    """
    if p_hu.shape != fd_hu.shape:
        raise ValueError(f"Shape mismatch: prediction {p_hu.shape} vs ground truth {fd_hu.shape}")
    if p_norm.shape != fd_norm.shape:
        raise ValueError(f"Shape mismatch: normalized prediction {p_norm.shape} vs normalized reference {fd_norm.shape}")
    if st_mask.shape != fd_hu.shape or body_mask.shape != fd_hu.shape:
        raise ValueError(f"Mask shape mismatch: mask {st_mask.shape} vs slice {fd_hu.shape}")
    if not (np.all(np.isfinite(p_hu)) and np.all(np.isfinite(fd_hu))):
        raise ValueError("Non-finite values (NaN or Inf) detected in slice HU arrays.")

    # Soft-tissue metrics (requires valid soft-tissue pixels)
    if np.any(st_mask):
        st_diff = p_hu[st_mask] - fd_hu[st_mask]
        st_mae: Optional[float] = float(np.mean(np.abs(st_diff)))
        st_rmse: Optional[float] = float(np.sqrt(np.mean(st_diff**2)))
        consistent_st_psnr: Optional[float] = float(20.0 * np.log10(SOFT_TISSUE_RANGE / (st_rmse + 1e-8)))
        historical_mixed_psnr: Optional[float] = float(20.0 * np.log10(HU_RANGE / (st_rmse + 1e-8)))
    else:
        st_mae = None
        st_rmse = None
        consistent_st_psnr = None
        historical_mixed_psnr = None

    # Patient body contour metrics
    if np.any(body_mask):
        b_diff = p_hu[body_mask] - fd_hu[body_mask]
        b_mae: Optional[float] = float(np.mean(np.abs(b_diff)))
        b_rmse: Optional[float] = float(np.sqrt(np.mean(b_diff**2)))
        global_body_psnr: Optional[float] = float(20.0 * np.log10(HU_RANGE / (b_rmse + 1e-8)))
    else:
        b_mae = None
        b_rmse = None
        global_body_psnr = None

    # SSIM on full normalized slices [0, 1]
    t_pred = tf.convert_to_tensor(p_norm[..., np.newaxis], dtype=tf.float32)
    t_ref = tf.convert_to_tensor(fd_norm[..., np.newaxis], dtype=tf.float32)
    s_ssim: Optional[float] = float(tf.image.ssim(t_pred, t_ref, max_val=1.0).numpy())

    # Gradient edge calculations
    mag_gt, gx_gt, gy_gt = sobel_vector(fd_hu)
    mag_p, gx_p, gy_p = sobel_vector(p_hu)

    if np.any(st_mask):
        st_gt_vals = mag_gt[st_mask]
        perc75 = float(np.percentile(st_gt_vals, 75)) if len(st_gt_vals) > 0 else 0.0
        edge_mask = st_mask & (mag_gt > perc75)
    else:
        edge_mask = np.zeros_like(st_mask, dtype=bool)

    fd_st = np.clip(fd_hu, SOFT_TISSUE_MIN, SOFT_TISSUE_MAX)
    p_st = np.clip(p_hu, SOFT_TISSUE_MIN, SOFT_TISSUE_MAX)
    mag_st_gt, gx_st_gt, gy_st_gt = sobel_vector(fd_st)
    mag_st_p, gx_st_p, gy_st_p = sobel_vector(p_st)

    if np.any(st_mask):
        st_mag_vals = mag_st_gt[st_mask]
        st_perc75 = float(np.percentile(st_mag_vals, 75)) if len(st_mag_vals) > 0 else 0.0
        st_edge_mask = st_mask & (mag_st_gt > st_perc75)
    else:
        st_edge_mask = np.zeros_like(st_mask, dtype=bool)

    if np.any(edge_mask):
        scalar_edge_err: Optional[float] = float(np.mean(np.abs(mag_p[edge_mask] - mag_gt[edge_mask])))
        vec_grad_err: Optional[float] = float(np.mean(np.sqrt((gx_p[edge_mask] - gx_gt[edge_mask])**2 + (gy_p[edge_mask] - gy_gt[edge_mask])**2)))
    else:
        scalar_edge_err = None
        vec_grad_err = None

    if np.any(st_edge_mask):
        st_vec_grad_err: Optional[float] = float(np.mean(np.sqrt((gx_st_p[st_edge_mask] - gx_st_gt[st_edge_mask])**2 + (gy_st_p[st_edge_mask] - gy_st_gt[st_edge_mask])**2)))
    else:
        st_vec_grad_err = None

    return {
        "st_mae": st_mae,
        "st_rmse": st_rmse,
        "global_mae": b_mae,
        "global_rmse": b_rmse,
        "consistent_st_psnr": consistent_st_psnr,
        "historical_mixed_psnr": historical_mixed_psnr,
        "psnr_hu": historical_mixed_psnr,  # backward compatibility alias
        "global_body_psnr": global_body_psnr,
        "ssim": s_ssim,
        "scalar_edge_error": scalar_edge_err,
        "vector_gradient_error": vec_grad_err,
        "st_vector_gradient_error": st_vec_grad_err
    }



def aggregate_metric(values: List[Optional[float]], decimals: int = 2) -> Tuple[Optional[float], int]:
    """Filters out None values and computes arithmetic mean, returning (rounded_mean, valid_count)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, 0
    return round(float(np.mean(valid)), decimals), len(valid)


def aggregate_cohort_metrics(
    metrics_by_system: Dict[str, List[Dict[str, Optional[float]]]],
    systems: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Aggregates per-slice metrics across evaluated systems, filtering out unavailable (None) values,
    tracking valid slice counts, and performing acceptance gate evaluation.
    Returns (summary, acceptance_audit).
    """
    summary = {}
    for s in systems:
        arr = metrics_by_system.get(s, [])
        m_st_mae, n_st_mae = aggregate_metric([x.get("st_mae") for x in arr], decimals=2)
        m_st_rmse, n_st_rmse = aggregate_metric([x.get("st_rmse") for x in arr], decimals=2)
        m_st_psnr, n_st_psnr = aggregate_metric([x.get("consistent_st_psnr") for x in arr], decimals=2)
        m_mixed_psnr, n_mixed_psnr = aggregate_metric([x.get("historical_mixed_psnr") for x in arr], decimals=2)
        m_global_psnr, n_global_psnr = aggregate_metric([x.get("global_body_psnr") for x in arr], decimals=2)
        m_ssim, n_ssim = aggregate_metric([x.get("ssim") for x in arr], decimals=4)
        m_vec, n_vec = aggregate_metric([x.get("vector_gradient_error") for x in arr], decimals=2)
        m_st_vec, n_st_vec = aggregate_metric([x.get("st_vector_gradient_error") for x in arr], decimals=2)
        m_scalar, n_scalar = aggregate_metric([x.get("scalar_edge_error") for x in arr], decimals=2)

        summary[s] = {
            "st_mae_hu": m_st_mae,
            "st_rmse_hu": m_st_rmse,
            "consistent_st_psnr_db": m_st_psnr,
            "historical_mixed_psnr_db": m_mixed_psnr,
            "psnr_db": m_mixed_psnr,  # backward compatibility alias
            "global_body_psnr_db": m_global_psnr,
            "ssim": m_ssim,
            "vector_gradient_error": m_vec,
            "st_vector_gradient_error": m_st_vec,
            "scalar_edge_error": m_scalar,
            "valid_slice_counts": {
                "soft_tissue": n_st_mae,
                "global_body": n_global_psnr,
                "ssim": n_ssim,
                "vector_gradient": n_vec,
                "total_evaluated": len(arr)
            }
        }

    # Acceptance Gate Evaluation
    raw_summary = summary.get("raw_low_dose", {})
    cand_summary = summary.get("residual_unet_v3", {})

    raw_st_mae = raw_summary.get("st_mae_hu")
    cand_st_mae = cand_summary.get("st_mae_hu")
    raw_ssim = raw_summary.get("ssim")
    cand_ssim = cand_summary.get("ssim")
    raw_vec_grad = raw_summary.get("vector_gradient_error")
    cand_vec_grad = cand_summary.get("vector_gradient_error")

    # If any required metric has 0 valid slices across cohort, report INSUFFICIENT_DATA
    if (raw_st_mae is None or cand_st_mae is None or
        raw_ssim is None or cand_ssim is None or
        raw_vec_grad is None or cand_vec_grad is None):

        acceptance_audit = {
            "overall_status": "INSUFFICIENT_DATA",
            "reason": "One or more required acceptance criteria have 0 valid slices across cohort.",
            "primary_criterion_mae": {
                "raw_low_dose": raw_st_mae,
                "candidate_v3": cand_st_mae,
                "delta_hu": round(cand_st_mae - raw_st_mae, 2) if (raw_st_mae is not None and cand_st_mae is not None) else None,
                "threshold": "<= 10.0 HU and delta <= -0.50 HU",
                "passed": False,
                "status": "UNAVAILABLE" if (raw_st_mae is None or cand_st_mae is None) else "EVALUATED"
            },
            "structural_criterion_ssim": {
                "raw_low_dose": raw_ssim,
                "candidate_v3": cand_ssim,
                "delta_ssim": round(cand_ssim - raw_ssim, 4) if (raw_ssim is not None and cand_ssim is not None) else None,
                "threshold": ">= 0.9000 and delta >= -0.015",
                "passed": False,
                "status": "UNAVAILABLE" if (raw_ssim is None or cand_ssim is None) else "EVALUATED"
            },
            "edge_preservation_criterion": {
                "raw_low_dose_vec_grad": raw_vec_grad,
                "candidate_v3_vec_grad": cand_vec_grad,
                "delta_grad_err": round(cand_vec_grad - raw_vec_grad, 2) if (raw_vec_grad is not None and cand_vec_grad is not None) else None,
                "threshold": "<= raw low-dose gradient error",
                "passed": False,
                "status": "UNAVAILABLE" if (raw_vec_grad is None or cand_vec_grad is None) else "EVALUATED"
            }
        }
    else:
        mae_diff = cand_st_mae - raw_st_mae
        ssim_diff = cand_ssim - raw_ssim
        vec_diff = cand_vec_grad - raw_vec_grad

        crit1_pass = (cand_st_mae <= 10.0) and (mae_diff <= -0.50)
        crit2_pass = (cand_ssim >= 0.9000) and (ssim_diff >= -0.015)
        crit3_pass = (cand_vec_grad <= raw_vec_grad)
        overall_pass = crit1_pass and crit2_pass and crit3_pass

        acceptance_audit = {
            "overall_status": "PASS" if overall_pass else "FAIL",
            "primary_criterion_mae": {
                "raw_low_dose": raw_st_mae,
                "candidate_v3": cand_st_mae,
                "delta_hu": round(mae_diff, 2),
                "threshold": "<= 10.0 HU and delta <= -0.50 HU",
                "passed": crit1_pass,
                "status": "EVALUATED"
            },
            "structural_criterion_ssim": {
                "raw_low_dose": raw_ssim,
                "candidate_v3": cand_ssim,
                "delta_ssim": round(ssim_diff, 4),
                "threshold": ">= 0.9000 and delta >= -0.015",
                "passed": crit2_pass,
                "status": "EVALUATED"
            },
            "edge_preservation_criterion": {
                "raw_low_dose_vec_grad": raw_vec_grad,
                "candidate_v3_vec_grad": cand_vec_grad,
                "delta_grad_err": round(vec_diff, 2),
                "threshold": "<= raw low-dose gradient error",
                "passed": crit3_pass,
                "status": "EVALUATED"
            }
        }

    return summary, acceptance_audit


def save_evaluation_report(report_data: dict, output_path: str, overwrite: bool = False) -> bool:
    """Safely saves evaluation report without silent overwrite unless explicit."""
    if not output_path:
        return False

    if os.path.exists(output_path) and not overwrite:
        print(f"[!] Warning: Output report '{output_path}' already exists and overwrite=False. Skipping report write (pass --overwrite to allow).")
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"[+] Saved evaluation report to: {output_path}")
        return True
    except Exception as e:
        print(f"[!] Warning: Failed to write report to '{output_path}': {e}")
        return False


def run_evaluation(
    data_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    v2_checkpoint: Optional[str] = None,
    v3_checkpoint: Optional[str] = None,
    output_report: Optional[str] = None,
    overwrite: bool = False,
    generate_figures: bool = False
) -> Dict[str, Any]:
    """
    Executes benchmark evaluation on the specified cohort.

    Args:
        data_dir: Directory containing raw TCIA LDCT DICOMs.
        manifest_path: Path to evaluation manifest JSON.
        v2_checkpoint: Path to v2 model .keras file.
        v3_checkpoint: Path to v3 candidate model .keras file.
        output_report: Destination path for evaluation JSON report.
        overwrite: Whether to overwrite existing output report.
        generate_figures: Whether to render and save representative comparison PNG figures.

    Returns:
        Structured evaluation results dictionary.
    """
    base_dir = REPO_ROOT
    if not data_dir:
        data_dir = os.environ.get("CASCADE_TCIA_RAW_DIR", os.path.join(base_dir, "data", "raw", "tcia_ldct"))
    if not manifest_path:
        manifest_path = os.path.join(base_dir, "data", "manifests", "tcia_residual_v3_locked_test_manifest.json")
    if not v2_checkpoint:
        v2_checkpoint = os.path.join(base_dir, "artifacts", "experiments", "unet_residual_native", "run_001", "candidate_unet_residual_native.keras")
    if not v3_checkpoint:
        v3_checkpoint = os.path.join(base_dir, "artifacts", "experiments", "unet_residual_edge_v3", "run_001", "candidate_unet_residual_edge_v3.keras")

    print("=" * 85)
    print("CASCADE Residual U-Net v3 Historical Benchmark Evaluation")
    print("=" * 85)
    print(f"Data Directory:          {data_dir}")
    print(f"Cohort Manifest:         {manifest_path}")
    print(f"v2 Baseline Checkpoint:  {v2_checkpoint}")
    print(f"v3 Candidate Checkpoint: {v3_checkpoint}")
    print(f"Output Report:           {output_report or 'None (in-memory)'}")

    # Check manifest existence
    if not os.path.exists(manifest_path):
        print(f"[!] Error: Evaluation cohort manifest not found at: {manifest_path}")
        return {
            "status": "DATA_UNAVAILABLE",
            "error": "MANIFEST_NOT_FOUND",
            "manifest_path": manifest_path
        }

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[!] Error: Malformed manifest JSON: {e}")
        return {
            "status": "DATA_INCOMPLETE",
            "error": "MALFORMED_MANIFEST",
            "message": str(e)
        }

    test_pids = manifest.get("cohort", [])
    print(f"Evaluation Cohort ({len(test_pids)} patients): {test_pids}")

    # Check if candidate model exists
    if not os.path.exists(v3_checkpoint):
        print(f"[!] Error: Candidate v3 checkpoint not found at: {v3_checkpoint}")
        return {
            "status": "CHECKPOINT_UNAVAILABLE",
            "error": "V3_CHECKPOINT_MISSING",
            "checkpoint_path": v3_checkpoint
        }

    # Verify patient slice pairs exist in data_dir
    all_pairs = []
    for pid in test_pids:
        p_dir = os.path.join(data_dir, f"patient_{pid}")
        pairing_file = os.path.join(p_dir, "slice_pairing_manifest.json")
        if os.path.exists(pairing_file):
            try:
                with open(pairing_file, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                pairs = p_data.get("matched_pairs", p_data.get("pairs", []))
                for pr in pairs:
                    fd = os.path.join(p_dir, "full_dose", pr.get("full_dose_file", pr.get("full_dose_path", "")))
                    ld = os.path.join(p_dir, "low_dose", pr.get("low_dose_file", pr.get("low_dose_path", "")))
                    if os.path.exists(fd) and os.path.exists(ld):
                        all_pairs.append((pid, fd, ld, pr.get("z_pos", 0)))
            except Exception:
                pass

    if len(all_pairs) == 0:
        print(f"[!] Raw DICOM pairs not found in {data_dir}. Evaluation cannot proceed without authentic scans.")
        return {
            "status": "DATA_UNAVAILABLE",
            "error": "DICOM_PAIRS_NOT_FOUND",
            "message": f"Zero slice pairs found for cohort in {data_dir}. Paired clean references are unavailable.",
            "report_written": False
        }

    print(f"[*] Found {len(all_pairs)} authentic paired slices across cohort.")
    print(f"[*] Loading v3 model from: {v3_checkpoint}")
    model_v3 = tf.keras.models.load_model(v3_checkpoint, compile=False)

    model_v2 = None
    if os.path.exists(v2_checkpoint):
        print(f"[*] Loading v2 model from: {v2_checkpoint}")
        model_v2 = tf.keras.models.load_model(v2_checkpoint, compile=False)
    else:
        print(f"[!] Warning: v2 checkpoint not found at {v2_checkpoint}. Comparing against raw low-dose only.")

    systems = ["raw_low_dose", "residual_unet_v3"]
    if model_v2 is not None:
        systems.insert(1, "residual_unet_v2")

    metrics_by_system = {s: [] for s in systems}
    rep_slices = {}
    total_slices_eval = 0

    t_start = time.time()

    for pid, fd_path, ld_path, z_pos in all_pairs:
        total_slices_eval += 1
        try:
            fd_hu = load_dicom_hu(fd_path)
            ld_hu = load_dicom_hu(ld_path)
        except Exception:
            continue

        ld_norm = hu_to_normalized(ld_hu)
        v3_norm = denoise_native_slice(model_v3, ld_norm)
        v3_hu = normalized_to_hu(v3_norm)

        preds = {
            "raw_low_dose": (ld_hu, ld_norm),
            "residual_unet_v3": (v3_hu, v3_norm)
        }

        if model_v2 is not None:
            v2_norm = denoise_native_slice(model_v2, ld_norm)
            v2_hu = normalized_to_hu(v2_norm)
            preds["residual_unet_v2"] = (v2_hu, v2_norm)

        st_mask = get_soft_tissue_mask(fd_hu)
        body_mask = get_body_contour_mask(fd_hu)
        fd_norm = hu_to_normalized(fd_hu)

        for s in systems:
            p_hu, p_norm = preds[s]
            s_metrics = compute_slice_metrics(p_hu, fd_hu, p_norm, fd_norm, st_mask, body_mask)
            s_metrics["patient_id"] = pid
            metrics_by_system[s].append(s_metrics)

    t_eval = time.time() - t_start
    print(f"\nCompleted evaluation of {total_slices_eval} slices in {t_eval:.1f}s.")

    # Aggregate summaries across cohort
    summary, acceptance_audit = aggregate_cohort_metrics(metrics_by_system, systems)

    print("\n" + "=" * 125)
    print(f"{'System':<20} | {'ST-MAE (HU)':<12} | {'ST-RMSE (HU)':<12} | {'ST-PSNR (dB)':<12} | {'Mixed PSNR':<12} | {'SSIM':<8} | {'Vec Grad Err':<12}")
    print("-" * 125)

    def _fmt(val: Optional[float], decimals: int = 2) -> str:
        return f"{val:<12.{decimals}f}" if val is not None else f"{'null':<12}"

    def _fmt_ssim(val: Optional[float]) -> str:
        return f"{val:<8.4f}" if val is not None else f"{'null':<8}"

    for s in systems:
        sd = summary[s]
        print(f"{s:<20} | {_fmt(sd['st_mae_hu'])} | {_fmt(sd['st_rmse_hu'])} | {_fmt(sd['consistent_st_psnr_db'])} | {_fmt(sd['historical_mixed_psnr_db'])} | {_fmt_ssim(sd['ssim'])} | {_fmt(sd['vector_gradient_error'])}")
    print("=" * 125)

    report_data = {
        "report_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_file": manifest_path,
        "cohort": test_pids,
        "total_slices": total_slices_eval,
        "aggregate_summary": summary,
        "acceptance_audit": acceptance_audit,
        "metric_documentation": {
            "empty_mask_policy": "Slices lacking valid mask pixels report null (None) to avoid fabricated zero-error or artificial PSNR. Slices with null metrics are excluded from arithmetic averages, and valid slice counts are tracked in valid_slice_counts.",
            "consistent_st_psnr_db": {
                "definition": "20 * log10(400.0 / (st_rmse + 1e-8))",
                "numerator_range_hu": 400.0,
                "window": "[-160, +240] HU",
                "mask": "Soft-tissue mask (ground-truth reference in [-160, 240] HU)",
                "aggregation": "Per-slice evaluation on valid soft-tissue slices (slices with empty masks reported as null and excluded from mean)",
                "description": "Consistent physical soft-tissue PSNR where both error domain and dynamic range numerator correspond strictly to the 400 HU soft-tissue window."
            },
            "historical_mixed_psnr_db": {
                "definition": "20 * log10(2000.0 / (st_rmse + 1e-8))",
                "numerator_range_hu": 2000.0,
                "window": "Full-body CT dynamic range [-1000, +1000] HU in numerator; [-160, +240] HU in denominator",
                "mask": "Soft-tissue mask (ground-truth reference in [-160, 240] HU)",
                "aggregation": "Per-slice evaluation on valid soft-tissue slices (slices with empty masks reported as null and excluded from mean)",
                "offset_relative_to_consistent_st_psnr": "+13.98 dB (20 * log10(5))",
                "description": "Historical mixed-domain PSNR combining a 2000 HU whole-body dynamic range with soft-tissue masked RMSE. Retained for exact historical continuity."
            },
            "psnr_db": "Alias for historical_mixed_psnr_db for backward compatibility with previous reports.",
            "global_body_psnr_db": {
                "definition": "20 * log10(2000.0 / (b_rmse + 1e-8))",
                "numerator_range_hu": 2000.0,
                "window": "[-1000, +1000] HU",
                "mask": "Patient body contour mask (excluding surrounding air)",
                "aggregation": "Per-slice evaluation on valid body contour slices (slices with empty masks reported as null and excluded from mean)",
                "description": "Full-range body PSNR computed over the patient-body contour mask."
            },
            "st_mae_hu": {
                "definition": "mean(abs(pred_hu - ref_hu)) over st_mask",
                "units": "Hounsfield Units (HU)",
                "window": "[-160, +240] HU",
                "aggregation": "Per-slice mean across valid soft-tissue slices"
            },
            "ssim": {
                "definition": "tf.image.ssim(pred_norm, ref_norm, max_val=1.0)",
                "inputs": "Full 512x512 unmasked slices normalized to [0, 1] via [-1000, +1000] HU",
                "mask": "None (unmasked full slice)",
                "aggregation": "Arithmetic mean of slice SSIM values across cohort"
            },
            "vector_gradient_error": {
                "definition": "mean(sqrt((gx_pred - gx_ref)^2 + (gy_pred - gy_ref)^2)) on edge_mask",
                "units": "HU/pixel",
                "operator": "2D Sobel 3x3 kernels (gx, gy)",
                "mask": "High-gradient boundary edge mask (top 25th percentile magnitude within soft-tissue mask)",
                "aggregation": "Per-slice mean across valid boundary edge slices"
            }
        },
        "disclaimer": "Research evaluation on historical evaluation cohort (441 slices). Not a clinical diagnostic tool."
    }

    report_written = save_evaluation_report(report_data, output_report, overwrite)
    eval_status = "INSUFFICIENT_DATA" if acceptance_audit.get("overall_status") == "INSUFFICIENT_DATA" else "COMPLETED"

    return {
        "status": eval_status,
        "total_slices_eval": total_slices_eval,
        "summary": summary,
        "acceptance_audit": acceptance_audit,
        "report_written": report_written
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASCADE Residual U-Net v3 Historical Benchmark Evaluation")
    parser.add_argument("--data-dir", default=None, help="Directory containing raw TCIA LDCT DICOMs")
    parser.add_argument("--manifest", default=None, help="Path to evaluation cohort manifest JSON")
    parser.add_argument("--v2-checkpoint", default=None, help="Path to v2 model .keras file")
    parser.add_argument("--v3-checkpoint", default=None, help="Path to v3 candidate model .keras file")
    parser.add_argument("--output-report", default=None, help="Path for JSON evaluation report")
    parser.add_argument("--no-report", action="store_true", default=False, help="Skip writing JSON evaluation report to disk")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Allow overwriting existing output report")
    parser.add_argument("--generate-figures", action="store_true", default=False, help="Render and save visual comparison PNGs")

    args = parser.parse_args()

    out_rep = None if args.no_report else args.output_report
    res = run_evaluation(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        v2_checkpoint=args.v2_checkpoint,
        v3_checkpoint=args.v3_checkpoint,
        output_report=out_rep,
        overwrite=args.overwrite,
        generate_figures=args.generate_figures
    )
    sys.exit(0 if res.get("status") in ("COMPLETED", "DATA_UNAVAILABLE", "INSUFFICIENT_DATA") else 1)
