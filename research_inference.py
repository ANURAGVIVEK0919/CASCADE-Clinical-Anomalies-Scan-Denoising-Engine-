"""
CASCADE Residual U-Net v3 Dedicated Research Inference Service

Implements an isolated, hardened research inference path for physical
CT DICOM slices using the accepted Residual U-Net v3 candidate checkpoint.

Security & Safety Guarantees:
1. Strict Calibrated Input Only:
   - Validates that inputs are genuine CT DICOM files (Modality == 'CT').
   - Rejects multiframe or 3D volumes; requires single 2D axial slices.
   - Requires finite, non-zero RescaleSlope and finite RescaleIntercept.
   - Enforces maximum upload size bound (32 MB).
   - Explicitly rejects ordinary 8-bit PNG/JPEG/BMP images and non-calibrated files.
2. Selected Metadata Redaction & Ephemeral Storage Policy:
   - Excludes all direct patient identifiers (PatientName, PatientBirthDate, PatientID), physician/operator names, institution fields, accession numbers, study/series/SOP UIDs, acquisition dates, and private tags from output payloads.
   - Preserves non-identifying physical CT calibration (RescaleSlope, RescaleIntercept) and spatial geometry (PixelSpacing, SliceThickness, SliceLocation) essential for accurate HU reconstruction.
   - Maintains strict ephemeral processing: uploaded byte streams are processed entirely in memory with zero persistent scan storage on disk.
   - Privacy Caveat: Formal DICOM de-identification (e.g. DICOM PS 3.15 Annex E Basic Application Level Confidentiality Profile) is NOT certified. Burned-in pixel annotations (if present in the input image) are rendered into 8-bit display previews without OCR/pixel scrubbing. Absolute claims of "zero PHI" or "HIPAA compliance" are avoided.
   - No raw DICOM metadata, pixel data, or patient identifiers are logged to console, disk, or telemetry.
3. Mathematically Consistent Preprocessing & Reconstruction:
   - Fixed, deterministic linear mapping: [-1000.0, +1000.0] HU -> [0.0, 1.0].
   - Native-resolution overlapping patch extraction (128x128 patches, stride 64).
   - Smooth 2D Hann window blending to eliminate seam artifacts.
   - Exact inverse mapping: [0.0, 1.0] -> [-1000.0, +1000.0] HU.
4. Non-Interference & Operational Safety:
   - Preserves source DICOM metadata and geometry without modification.
   - Never overwrites original scan files.
   - Strictly labeled as PRELIMINARY_RESEARCH_CANDIDATE / NOT FOR CLINICAL USE.
   - Separate telemetry metrics tracked independently from legacy models.
"""

import os
import io
import time
import base64
import hashlib
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from models.residual_unet import (
    hu_to_normalized,
    normalized_to_hu,
    extract_grid_patches,
    make_2d_hann_window,
    blend_grid_patches,
    get_body_contour_mask,
    get_soft_tissue_mask,
    HU_MIN,
    HU_MAX,
    HU_RANGE,
    SOFT_TISSUE_MIN,
    SOFT_TISSUE_MAX,
    BODY_MASK_THRESHOLD
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
V3_CHECKPOINT_REL = os.path.join(
    "artifacts", "experiments", "unet_residual_edge_v3", "run_001", "candidate_unet_residual_edge_v3.keras"
)
V3_CHECKPOINT_PATH = os.path.join(REPO_ROOT, V3_CHECKPOINT_REL)
EXPECTED_V3_SHA256 = "bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea"

# Maximum allowable input payload size (32 MB)
MAX_DICOM_UPLOAD_SIZE = 32 * 1024 * 1024

# Clinical display window definitions: (window_min, window_max)
DISPLAY_WINDOWS: Dict[str, Tuple[float, float]] = {
    "soft_tissue": (-160.0, 240.0),   # Window Width 400, Level 40
    "lung": (-1000.0, 200.0),         # Window Width 1200, Level -400
    "bone": (-100.0, 900.0),          # Window Width 1000, Level 400
    "full_body": (-1000.0, 1000.0)    # Window Width 2000, Level 0
}

_CACHED_V3_MODEL = None

# Telemetry counters for the isolated research inference pipeline
_RESEARCH_METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "last_latency_ms": None,
    "last_status": "IDLE"
}


def record_research_request(success: bool, duration_ms: float, status: str = "COMPLETED"):
    """Records research endpoint request telemetry."""
    _RESEARCH_METRICS["requests_total"] += 1
    if success:
        _RESEARCH_METRICS["requests_success"] += 1
    else:
        _RESEARCH_METRICS["requests_failed"] += 1
    _RESEARCH_METRICS["last_latency_ms"] = round(duration_ms, 2)
    _RESEARCH_METRICS["last_status"] = status


def get_research_telemetry() -> Dict[str, Any]:
    """Returns a snapshot of research inference telemetry."""
    return dict(_RESEARCH_METRICS)


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def verify_checkpoint_integrity(ckpt_path: str = V3_CHECKPOINT_PATH) -> bool:
    """Verifies that the candidate checkpoint exists and matches its expected SHA-256."""
    if not os.path.exists(ckpt_path):
        return False
    try:
        actual_hash = compute_file_sha256(ckpt_path)
        return actual_hash == EXPECTED_V3_SHA256
    except Exception:
        return False


def get_research_v3_model(ckpt_path: str = V3_CHECKPOINT_PATH):
    """
    Loads and caches the Residual U-Net v3 model.
    Enforces integrity check to prevent running corrupted or modified weights.
    """
    global _CACHED_V3_MODEL
    if _CACHED_V3_MODEL is not None and ckpt_path == V3_CHECKPOINT_PATH:
        return _CACHED_V3_MODEL

    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to load Residual U-Net v3.")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Residual U-Net v3 checkpoint weights file not found on server.")

    actual_hash = compute_file_sha256(ckpt_path)
    if actual_hash != EXPECTED_V3_SHA256:
        raise ValueError(
            f"Checkpoint integrity violation! Expected SHA-256 {EXPECTED_V3_SHA256}, got {actual_hash}"
        )

    loaded_model = tf.keras.models.load_model(ckpt_path, compile=False)
    if ckpt_path == V3_CHECKPOINT_PATH:
        _CACHED_V3_MODEL = loaded_model
    return loaded_model


def validate_and_load_dicom(
    file_source: Union[str, bytes, io.BytesIO]
) -> Tuple[np.ndarray, Dict[str, Any], Any]:
    """
    Validates that the input is a genuine CT DICOM with valid calibration metadata.
    Enforces bounds, single 2D axial slice, and strict non-PHI extraction.

    Returns:
        hu_pixels: 2D numpy array (H, W) in physical Hounsfield Units.
        meta: Dictionary of non-identifying DICOM metadata and geometry.
        ds: The parsed pydicom FileDataset.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for CT DICOM research inference.")

    # 1. Bounded size verification and byte parsing
    if isinstance(file_source, bytes):
        if len(file_source) == 0:
            raise ValueError("Empty DICOM file received.")
        if len(file_source) > MAX_DICOM_UPLOAD_SIZE:
            raise ValueError(f"Payload exceeds maximum allowable size limit of {MAX_DICOM_UPLOAD_SIZE // (1024*1024)} MB.")

        # Reject standard 8-bit image magic headers early
        if file_source.startswith(b"\x89PNG") or file_source.startswith(b"\xff\xd8\xff") or file_source.startswith(b"GIF") or file_source.startswith(b"BM"):
            raise ValueError(
                "Unsupported input format: Ordinary 8-bit image detected. "
                "Residual U-Net v3 requires calibrated CT DICOM slices with physical Hounsfield Units."
            )
        buffer = io.BytesIO(file_source)
        try:
            ds = pydicom.dcmread(buffer)
        except Exception:
            raise ValueError("Invalid or corrupt DICOM file: unable to parse data stream.")
    elif isinstance(file_source, io.BytesIO):
        file_source.seek(0, io.SEEK_END)
        buf_size = file_source.tell()
        file_source.seek(0)
        if buf_size == 0:
            raise ValueError("Empty DICOM file received.")
        if buf_size > MAX_DICOM_UPLOAD_SIZE:
            raise ValueError(f"Payload exceeds maximum allowable size limit of {MAX_DICOM_UPLOAD_SIZE // (1024*1024)} MB.")
        try:
            ds = pydicom.dcmread(file_source)
        except Exception:
            raise ValueError("Invalid or corrupt DICOM buffer: unable to parse data stream.")
    elif isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError("Specified DICOM file not found.")
        file_size = os.path.getsize(file_source)
        if file_size == 0:
            raise ValueError("Empty DICOM file received.")
        if file_size > MAX_DICOM_UPLOAD_SIZE:
            raise ValueError(f"Payload exceeds maximum allowable size limit of {MAX_DICOM_UPLOAD_SIZE // (1024*1024)} MB.")
        lower = file_source.lower()
        if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".bmp"):
            raise ValueError(
                "Unsupported input format: Ordinary 8-bit image detected. "
                "Residual U-Net v3 requires calibrated CT DICOM slices with physical Hounsfield Units."
            )
        try:
            ds = pydicom.dcmread(file_source)
        except Exception:
            raise ValueError("Invalid or corrupt DICOM file: unable to parse data stream.")
    else:
        raise TypeError("Unsupported file_source input type.")

    # 2. Modality verification
    modality = str(ds.get("Modality", "")).upper()
    if modality != "CT":
        raise ValueError(
            f"Unsupported modality: '{modality}'. Residual U-Net v3 exclusively supports CT scans."
        )

    # 3. Multiframe and 3D rejection
    num_frames = getattr(ds, "NumberOfFrames", None)
    if num_frames is not None:
        try:
            val_frames = int(num_frames)
        except (ValueError, TypeError):
            val_frames = 1
        if val_frames > 1:
            raise ValueError("Multiframe DICOM files are not supported. Only single 2D axial CT slices are accepted.")


    # 4. Calibration parameters verification
    if "RescaleSlope" not in ds or "RescaleIntercept" not in ds:
        raise ValueError(
            "Missing required DICOM rescale metadata (RescaleSlope, RescaleIntercept). Cannot compute physical Hounsfield units."
        )

    try:
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
    except (ValueError, TypeError):
        raise ValueError("Invalid RescaleSlope or RescaleIntercept tag: values must be numeric.")

    if not (np.isfinite(slope) and np.isfinite(intercept)):
        raise ValueError("Invalid RescaleSlope or RescaleIntercept tag: non-finite values are not supported.")

    if slope == 0.0:
        raise ValueError("Invalid RescaleSlope tag: slope cannot be zero.")

    # 5. Pixel array extraction and dimension bounds
    try:
        raw_pixels = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to extract pixel array from DICOM: {e}")

    if len(raw_pixels.shape) != 2:
        raise ValueError(
            f"Unsupported dimensions: expected 2D single slice (H, W), got shape {raw_pixels.shape}."
        )

    rows, cols = raw_pixels.shape
    if rows < 128 or cols < 128:
        raise ValueError(f"Unsupported slice dimensions {raw_pixels.shape}. Minimum supported dimension is 128x128.")
    if rows > 2048 or cols > 2048:
        raise ValueError(f"Unsupported slice dimensions {raw_pixels.shape}. Maximum supported dimension is 2048x2048.")

    if not np.all(np.isfinite(raw_pixels)):
        raise ValueError("Corrupt pixel array: contains non-finite (NaN or Inf) values.")

    hu_pixels = raw_pixels * slope + intercept

    # 6. Selected Non-Identifying Metadata Extraction (Whitelisted Parameters)
    # Extracts only numeric calibration and geometry required for scientific interpretation.
    # Patient identifiers, physician fields, institution data, study/series UIDs, and private tags
    # are strictly excluded from output payloads.
    img_pos = [float(x) for x in ds.get("ImagePositionPatient", [0.0, 0.0, 0.0])]
    pixel_spacing = [float(x) for x in ds.get("PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(ds.get("SliceThickness", 1.0))
    burned_in = str(ds.get("BurnedInAnnotation", "NO")).strip().upper()

    meta = {
        "patient_id": "[REDACTED_FOR_PRIVACY]",
        "modality": modality,
        "rows": rows,
        "columns": cols,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "z_position": round(img_pos[2], 2) if len(img_pos) >= 3 else 0.0,
        "pixel_spacing": [round(x, 4) for x in pixel_spacing],
        "slice_thickness": round(slice_thickness, 2),
        "hu_min": round(float(np.min(hu_pixels)), 1),
        "hu_max": round(float(np.max(hu_pixels)), 1),
        "hu_mean": round(float(np.mean(hu_pixels)), 1),
        "burned_in_annotation": burned_in
    }

    return hu_pixels, meta, ds


def hu_to_display_uint8(
    hu_array: np.ndarray,
    window_mode: str = "soft_tissue"
) -> np.ndarray:
    """
    Renders physical Hounsfield Units into an 8-bit display image using clinical windowing.
    """
    w_min, w_max = DISPLAY_WINDOWS.get(window_mode, DISPLAY_WINDOWS["soft_tissue"])
    clipped = np.clip(hu_array, w_min, w_max)
    normalized = (clipped - w_min) / (w_max - w_min)
    return (normalized * 255.0).astype(np.uint8)


def array_to_png_base64(img_uint8: np.ndarray) -> str:
    """Encodes a uint8 2D numpy array to a base64 PNG data string."""
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_bytes = base64.b64encode(buf.getvalue())
    return f"data:image/png;base64,{b64_bytes.decode('utf-8')}"


def denoise_native_slice_v3(
    model,
    low_dose_norm_512: np.ndarray,
    patch_size: int = 128,
    stride: int = 64
) -> np.ndarray:
    """
    Performs patch extraction, model prediction, and 2D Hann window blending
    on a full-resolution normalized slice.
    """
    patches, coords = extract_grid_patches(low_dose_norm_512, patch_size=patch_size, stride=stride)
    patches_input = tf.convert_to_tensor(np.expand_dims(patches, axis=-1), dtype=tf.float32)
    preds = model(patches_input, training=False).numpy()
    pred_patches = preds.squeeze(axis=-1)

    reconstructed = blend_grid_patches(
        pred_patches,
        coords,
        image_shape=low_dose_norm_512.shape[:2],
        patch_size=patch_size
    )
    return reconstructed


def denoise_dicom_research(
    input_source: Union[str, bytes, io.BytesIO],
    reference_source: Optional[Union[str, bytes, io.BytesIO]] = None,
    window_mode: str = "soft_tissue"
) -> Dict[str, Any]:
    """
    Executes the isolated research inference pipeline for calibrated CT DICOM slices.

    Returns structured analysis with raw HU, restored HU, difference maps,
    display renderings, non-identifying DICOM metadata, and scientific research disclaimers.
    """
    # 1. Validate and load input DICOM
    raw_hu, raw_meta, raw_ds = validate_and_load_dicom(input_source)

    # 2. Checkpoint and model retrieval
    model = get_research_v3_model()

    # 3. Fixed HU Normalization [-1000, +1000] HU -> [0, 1]
    norm_input = hu_to_normalized(raw_hu)

    # 4. Native resolution patch-based denoising with 2D Hann blending
    norm_denoised_raw = denoise_native_slice_v3(model, norm_input, patch_size=128, stride=64)

    # 5. Output-Range Accounting
    pred_norm_min = float(np.min(norm_denoised_raw))
    pred_norm_max = float(np.max(norm_denoised_raw))
    unclipped_hu_min = float(pred_norm_min * HU_RANGE + HU_MIN)
    unclipped_hu_max = float(pred_norm_max * HU_RANGE + HU_MIN)

    below_0_count = int(np.sum(norm_denoised_raw < 0.0))
    above_1_count = int(np.sum(norm_denoised_raw > 1.0))
    total_oor_count = below_0_count + above_1_count
    total_pixels = norm_denoised_raw.size

    # Domain-bounded clamping: Restricts normalized predictions to [0.0, 1.0]
    # Corresponding to the [-1000.0, +1000.0] HU calibration manifold.
    norm_denoised = np.clip(norm_denoised_raw, 0.0, 1.0)

    # 6. Exact inverse conversion back to physical HU
    denoised_hu = normalized_to_hu(norm_denoised)

    # 7. Physical HU residual difference map
    diff_hu = np.abs(denoised_hu - raw_hu)

    # 7. Render display images in the requested clinical window
    raw_display = hu_to_display_uint8(raw_hu, window_mode=window_mode)
    denoised_display = hu_to_display_uint8(denoised_hu, window_mode=window_mode)
    diff_display = np.clip((diff_hu / 100.0) * 255.0, 0, 255).astype(np.uint8)

    # 8. Optional Reference Benchmarking (if paired routine-dose scan is provided)
    metrics = None
    if reference_source is not None:
        try:
            ref_hu, ref_meta, _ = validate_and_load_dicom(reference_source)
            if ref_hu.shape == raw_hu.shape:
                st_mask = get_soft_tissue_mask(ref_hu)

                if np.sum(st_mask) > 100:
                    st_diff_raw = raw_hu[st_mask] - ref_hu[st_mask]
                    st_diff_v3 = denoised_hu[st_mask] - ref_hu[st_mask]

                    st_mae_raw = float(np.mean(np.abs(st_diff_raw)))
                    st_mae_v3 = float(np.mean(np.abs(st_diff_v3)))

                    st_rmse_raw = float(np.sqrt(np.mean(st_diff_raw**2)))
                    st_rmse_v3 = float(np.sqrt(np.mean(st_diff_v3**2)))

                    st_psnr_raw = float(20.0 * np.log10(400.0 / max(1e-6, st_rmse_raw)))
                    st_psnr_v3 = float(20.0 * np.log10(400.0 / max(1e-6, st_rmse_v3)))

                    metrics = {
                        "soft_tissue_mae_raw_hu": round(st_mae_raw, 2),
                        "soft_tissue_mae_v3_hu": round(st_mae_v3, 2),
                        "soft_tissue_mae_reduction_pct": round(((st_mae_raw - st_mae_v3) / st_mae_raw) * 100.0, 2),
                        "soft_tissue_rmse_raw_hu": round(st_rmse_raw, 2),
                        "soft_tissue_rmse_v3_hu": round(st_rmse_v3, 2),
                        "consistent_st_psnr_raw_db": round(st_psnr_raw, 2),
                        "consistent_st_psnr_v3_db": round(st_psnr_v3, 2),
                        "consistent_st_psnr_gain_db": round(st_psnr_v3 - st_psnr_raw, 2),
                        "evaluation_domain": "Abdominal Soft-Tissue [-160, 240] HU, Range 400.0 HU"
                    }
        except Exception:
            # Do not leak internal reference processing tracebacks
            metrics = {"reference_evaluation_error": "Failed to compute reference metrics for provided scan."}

    # 9. Assembly of safe research response payload
    return {
        "status": "SUCCESS",
        "pipeline": "research_dicom_pipeline",
        "model_version": "Residual U-Net v3 (candidate_unet_residual_edge_v3.keras)",
        "checkpoint_sha256": EXPECTED_V3_SHA256,
        "is_research_model": True,
        "research_classification": "PRELIMINARY_RESEARCH_CANDIDATE",
        "clinical_safety_notice": (
            "FOR INVESTIGATIONAL / RESEARCH USE ONLY. This system is a research candidate (Residual U-Net v3) "
            "and is NOT approved for clinical diagnosis, patient triage, or primary radiological review under "
            "FDA, CE, or regional medical device regulations. Denoised output must never be presented as an "
            "approved replacement for original clinical CT scans."
        ),
        "privacy_notice": (
            "Selected metadata redaction applied. The response payload contains only whitelisted geometric "
            "and intensity calibration parameters. Uploaded scans are processed ephemerally in-memory without "
            "persistent disk storage. Formal DICOM PS 3.15 de-identification is NOT certified; burned-in pixel "
            "annotations, if present in the acquired scan, are not scrubbed."
        ),
        "source_metadata": raw_meta,
        "window_mode": window_mode,
        "window_bounds_hu": DISPLAY_WINDOWS.get(window_mode, DISPLAY_WINDOWS["soft_tissue"]),
        "dimensions": {
            "rows": raw_hu.shape[0],
            "columns": raw_hu.shape[1],
            "patch_size": 128,
            "stride": 64,
            "blending": "2D Hann Window"
        },
        "statistics": {
            "raw_hu_min": raw_meta["hu_min"],
            "raw_hu_max": raw_meta["hu_max"],
            "raw_hu_mean": raw_meta["hu_mean"],
            "denoised_hu_min": round(float(np.min(denoised_hu)), 1),
            "denoised_hu_max": round(float(np.max(denoised_hu)), 1),
            "denoised_hu_mean": round(float(np.mean(denoised_hu)), 1),
            "mean_abs_diff_hu": round(float(np.mean(diff_hu)), 2),
            "max_abs_diff_hu": round(float(np.max(diff_hu)), 2)
        },
        "output_range_accounting": {
            "policy": "DOMAIN_BOUNDED_CLAMPING",
            "policy_rationale": (
                "Normalized predictions are clamped to the defined [0.0, 1.0] training domain "
                "([-1000.0, +1000.0] HU) to eliminate unphysical air extrapolation (< -1000 HU) "
                "and bone density overshoot (> +1000 HU) caused by unconstrained residual addition."
            ),
            "clamping_applied": bool(total_oor_count > 0),
            "valid_range_hu": [HU_MIN, HU_MAX],
            "raw_predictions": {
                "normalized_min": round(pred_norm_min, 6),
                "normalized_max": round(pred_norm_max, 6),
                "unclipped_hu_min": round(unclipped_hu_min, 1),
                "unclipped_hu_max": round(unclipped_hu_max, 1)
            },
            "out_of_range_pixels": {
                "below_min_count": below_0_count,
                "below_min_pct": round((below_0_count / total_pixels) * 100.0, 4),
                "above_max_count": above_1_count,
                "above_max_pct": round((above_1_count / total_pixels) * 100.0, 4),
                "total_oor_count": total_oor_count,
                "total_oor_pct": round((total_oor_count / total_pixels) * 100.0, 4)
            }
        },
        "display_images": {
            "raw_low_dose_base64": array_to_png_base64(raw_display),
            "denoised_v3_base64": array_to_png_base64(denoised_display),
            "residual_diff_base64": array_to_png_base64(diff_display)
        },
        "quantitative_metrics": metrics
    }
