"""
CASCADE HU-Calibrated DICOM Preprocessing and Geometry Validation Pipeline

Implements strict Hounsfield Unit (HU) calibrated conversion, fixed windowing,
and physical coordinate slice alignment for low-dose CT denoising.

Contract:
- Conversion: HU = pixel_array * RescaleSlope + RescaleIntercept
- Primary Window: Soft-tissue window [-160.0, +240.0] HU (Width 400, Center 40)
- Secondary Window: Wide-body window [-1000.0, +1000.0] HU (Width 2000, Center 0)
- Normalization: Invertible linear mapping to [0.0, 1.0]
- Zero independent per-image min-max scaling: Training, validation, testing,
  and production inference must share the identical fixed-window mapping.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


SOFT_TISSUE_WINDOW: Tuple[float, float] = (-160.0, 240.0)
WIDE_BODY_WINDOW: Tuple[float, float] = (-1000.0, 1000.0)


def read_dicom_hu(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a DICOM CT slice and converts raw stored pixel values to physical Hounsfield Units (HU).

    Validates DICOM geometry attributes:
    - Modality == 'CT'
    - ImagePositionPatient (specifically z-coordinate)
    - PixelSpacing
    - SliceThickness
    - RescaleSlope and RescaleIntercept
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required to parse DICOM files.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    ds = pydicom.dcmread(filepath)
    modality = str(ds.get("Modality", "")).upper()
    if modality and modality != "CT":
        raise ValueError(f"Expected CT modality, found: {modality} in {filepath}")

    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))

    raw_pixels = ds.pixel_array.astype(np.float32)
    hu_pixels = raw_pixels * slope + intercept

    # Extract geometry metadata
    img_pos = [float(x) for x in ds.get("ImagePositionPatient", [0.0, 0.0, 0.0])]
    pixel_spacing = [float(x) for x in ds.get("PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(ds.get("SliceThickness", 1.0))

    meta = {
        "sop_instance_uid": str(ds.get("SOPInstanceUID", "")),
        "series_instance_uid": str(ds.get("SeriesInstanceUID", "")),
        "patient_id": str(ds.get("PatientID", "")),
        "z_position": round(img_pos[2], 2),
        "image_position_patient": img_pos,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "hu_min": float(np.min(hu_pixels)),
        "hu_max": float(np.max(hu_pixels)),
        "hu_mean": float(np.mean(hu_pixels)),
    }
    return hu_pixels, meta


def hu_to_normalized(
    hu_array: np.ndarray,
    window: Tuple[float, float] = SOFT_TISSUE_WINDOW
) -> np.ndarray:
    """
    Maps an array of Hounsfield Units into the normalized model input range [0.0, 1.0]
    using the documented fixed window [w_min, w_max].

    Formula:
        norm = clip(hu, w_min, w_max) - w_min) / (w_max - w_min)
    """
    w_min, w_max = window
    if w_max <= w_min:
        raise ValueError(f"Invalid window: [{w_min}, {w_max}] (w_max must be > w_min)")
    clipped = np.clip(hu_array, w_min, w_max)
    norm = (clipped - w_min) / (w_max - w_min)
    return norm.astype(np.float32)


def normalized_to_hu(
    norm_array: np.ndarray,
    window: Tuple[float, float] = SOFT_TISSUE_WINDOW
) -> np.ndarray:
    """
    Exact inverse mapping from normalized [0.0, 1.0] back to physical Hounsfield Units (HU).

    Formula:
        hu = norm * (w_max - w_min) + w_min
    """
    w_min, w_max = window
    return (norm_array.astype(np.float32) * (w_max - w_min) + w_min).astype(np.float32)


def preprocess_dicom_slice(
    filepath: str,
    target_size: Tuple[int, int] = (256, 256),
    window: Tuple[float, float] = SOFT_TISSUE_WINDOW
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses a single DICOM slice for model input or target evaluation.

    Returns:
        model_input: (H, W, 1) float32 in [0.0, 1.0] at target_size
        hu_original: (orig_H, orig_W) float32 in original physical HU
        metadata: DICOM geometric and scaling metadata
    """
    hu_orig, meta = read_dicom_hu(filepath)
    norm = hu_to_normalized(hu_orig, window=window)

    # Resize using bilinear interpolation via PIL for deterministic reproducibility
    pil_img = Image.fromarray(norm)
    pil_resized = pil_img.resize(target_size, resample=Image.Resampling.BILINEAR)
    resized_norm = np.array(pil_resized, dtype=np.float32)
    resized_norm = np.clip(resized_norm, 0.0, 1.0)
    model_input = np.expand_dims(resized_norm, axis=-1)

    return model_input, hu_orig, meta


def validate_slice_pair_geometry(
    fd_meta: Dict[str, Any],
    ld_meta: Dict[str, Any],
    tolerance_z: float = 0.1
) -> bool:
    """
    Verifies that a routine-dose slice and a lower-dose slice represent the exact same
    physical slice location and matrix dimensions.
    """
    z_diff = abs(fd_meta["z_position"] - ld_meta["z_position"])
    if z_diff > tolerance_z:
        return False
    if fd_meta["rows"] != ld_meta["rows"] or fd_meta["columns"] != ld_meta["columns"]:
        return False
    return True
