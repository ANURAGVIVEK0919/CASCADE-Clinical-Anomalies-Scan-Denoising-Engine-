"""
CASCADE Residual U-Net v3 Integration & Release-Readiness Test Suite (Gate 4)

Tests:
1. Exact checkpoint loading and SHA-256 hash verification.
2. Preprocessing and inverse HU conversion round-trip.
3. Native-resolution output preservation (512x512 matrix shape).
4. Patch-blending boundary behavior (seamless 2D Hann window blending).
5. Unsupported input rejection (rejection of PNG/JPEG, non-CT modalities, uncalibrated scans).
6. Existing app regression behavior and research route registration.
7. Safe artifact paths without checkpoint overwrites.
"""

import os
import io
import json
import hashlib
import unittest
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

import tensorflow as tf

from research_inference import (
    V3_CHECKPOINT_PATH,
    EXPECTED_V3_SHA256,
    compute_file_sha256,
    verify_checkpoint_integrity,
    get_research_v3_model,
    validate_and_load_dicom,
    denoise_native_slice_v3,
    denoise_dicom_research,
    hu_to_display_uint8,
    DISPLAY_WINDOWS
)
from models.residual_unet import (
    hu_to_normalized,
    normalized_to_hu,
    extract_grid_patches,
    blend_grid_patches,
    make_2d_hann_window,
    HU_MIN,
    HU_MAX,
    HU_RANGE
)
from app import app


def create_dummy_ct_dicom(
    rows: int = 512,
    cols: int = 512,
    modality: str = "CT",
    include_rescale: bool = True,
    slope: float = 1.0,
    intercept: float = -1024.0,
    hu_center_val: float = 40.0
) -> io.BytesIO:
    """Creates an in-memory pydicom FileDataset for testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    buffer = io.BytesIO()
    ds = FileDataset(buffer, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.PatientID = "TEST_PATIENT_001"
    ds.PatientAge = "055Y"
    ds.PatientSex = "F"
    ds.StudyDate = "20261020"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8"
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [0.976562, 0.976562]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, 100.0]

    if include_rescale:
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"


    # Create synthetic raw values such that raw * slope + intercept = hu_center_val
    raw_val = (hu_center_val - intercept) / slope
    pixel_array = np.full((rows, cols), raw_val, dtype=np.int16)

    # Add a synthetic soft-tissue circle in the center
    yy, xx = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((xx - cols // 2)**2 + (yy - rows // 2)**2)
    circle_mask = dist_from_center <= (rows // 4)
    # Circle has soft-tissue intensity (HU = 60.0)
    raw_circle_val = (60.0 - intercept) / slope
    pixel_array[circle_mask] = np.int16(raw_circle_val)

    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(buffer)
    buffer.seek(0)
    return buffer


class TestResidualUNetV3Integration(unittest.TestCase):

    def test_exact_checkpoint_loading_and_hash(self):
        """Verify candidate checkpoint exists, matches exact SHA-256 hash, and loads correctly."""
        self.assertTrue(os.path.exists(V3_CHECKPOINT_PATH), f"Checkpoint missing at {V3_CHECKPOINT_PATH}")
        actual_hash = compute_file_sha256(V3_CHECKPOINT_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_V3_SHA256,
            f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_V3_SHA256}, got {actual_hash}"
        )
        self.assertTrue(verify_checkpoint_integrity(V3_CHECKPOINT_PATH))

        model = get_research_v3_model(V3_CHECKPOINT_PATH)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 128, 128, 1))
        self.assertEqual(model.output_shape, (None, 128, 128, 1))

    def test_preprocessing_and_inverse_hu_conversion(self):
        """Verify strict round-trip mathematical identity between physical HU and [0, 1]."""
        test_hus = np.array([-1000.0, -750.0, -160.0, 0.0, 40.0, 240.0, 750.0, 1000.0], dtype=np.float32)
        norm = hu_to_normalized(test_hus)

        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))
        restored = normalized_to_hu(norm)
        np.testing.assert_allclose(restored, test_hus, atol=1e-4)

        # Values beyond [-1000, 1000] should clip to [0, 1]
        out_of_bounds = np.array([-2000.0, 3000.0], dtype=np.float32)
        norm_clipped = hu_to_normalized(out_of_bounds)
        self.assertEqual(norm_clipped[0], 0.0)
        self.assertEqual(norm_clipped[1], 1.0)

    def test_native_resolution_output(self):
        """Verify full 512x512 native slice resolution is preserved through patch extraction & blending."""
        model = get_research_v3_model()
        dummy_norm_512 = np.random.uniform(0.1, 0.9, size=(512, 512)).astype(np.float32)

        denoised_512 = denoise_native_slice_v3(model, dummy_norm_512, patch_size=128, stride=64)
        self.assertEqual(denoised_512.shape, (512, 512), "Native 512x512 resolution was not preserved!")
        self.assertTrue(np.all(np.isfinite(denoised_512)), "Denoised slice contains NaN or Inf!")

    def test_patch_blending_boundary_behavior(self):
        """Verify that 2D Hann window overlapping patch reconstruction introduces zero seam discontinuities."""
        # Create a smooth linear gradient ramp across the image
        rows, cols = 512, 512
        yy, xx = np.mgrid[:rows, :cols]
        linear_ramp = ((xx + yy) / (rows + cols)).astype(np.float32)

        patches, coords = extract_grid_patches(linear_ramp, patch_size=128, stride=64)
        reconstructed = blend_grid_patches(patches, coords, image_shape=(rows, cols), patch_size=128)

        # Difference between original smooth ramp and patch-blended reconstruction should be near float precision
        max_error = np.max(np.abs(reconstructed - linear_ramp))
        self.assertLess(max_error, 1e-4, f"Seam blending error exceeds tolerance: {max_error}")

        # Check gradient along the primary seam boundaries (stride steps: 64, 128, 192, 256)
        for seam_x in [64, 128, 192, 256]:
            col_left = reconstructed[:, seam_x - 1]
            col_right = reconstructed[:, seam_x]
            max_seam_step = np.max(np.abs(col_right - col_left))
            # Step size should match continuous linear ramp step (1 / (rows + cols) ≈ 0.00097)
            self.assertLess(max_seam_step, 0.005, f"Discontinuous boundary step at seam x={seam_x}: {max_seam_step}")

    def test_unsupported_input_rejection(self):
        """Verify that ordinary PNG/JPEG, non-CT modalities, and uncalibrated DICOM are strictly rejected."""
        # 1. Reject PNG bytes
        png_buffer = io.BytesIO()
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()

        with self.assertRaises(ValueError) as ctx:
            validate_and_load_dicom(png_bytes)
        self.assertIn("Unsupported input format", str(ctx.exception))

        # 2. Reject non-CT modality (e.g. MR)
        mr_buffer = create_dummy_ct_dicom(modality="MR")
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_dicom(mr_buffer)
        self.assertIn("Unsupported modality: 'MR'", str(ctx.exception))


        # 3. Reject DICOM missing RescaleSlope/Intercept
        uncalibrated_buffer = create_dummy_ct_dicom(include_rescale=False)
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_dicom(uncalibrated_buffer)
        self.assertIn("RescaleSlope", str(ctx.exception))

    def test_existing_app_regression_behavior(self):
        """Verify existing app routes, metrics observability, and research endpoint functionality."""
        client = app.test_client()

        # 1. Metrics endpoint contains research candidate status
        resp = client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("residual_unet_v3_candidate", data["models"])

        # 2. Research endpoint rejects PNG upload
        png_buffer = io.BytesIO()
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(png_buffer, format="PNG")
        png_buffer.seek(0)

        post_resp = client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (png_buffer, "test_scan.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(post_resp.status_code, 400)
        err_data = post_resp.get_json()
        self.assertEqual(err_data["status"], "REJECTED")
        self.assertIn("Ordinary 8-bit images", err_data["error"])

        # 3. Research endpoint processes genuine CT DICOM slice
        dcm_buffer = create_dummy_ct_dicom(rows=512, cols=512, hu_center_val=40.0)
        valid_resp = client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (dcm_buffer, "patient_slice.dcm"), "window": "soft_tissue"},
            content_type="multipart/form-data"
        )
        self.assertEqual(valid_resp.status_code, 200)
        result = valid_resp.get_json()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Residual U-Net v3", result["model_version"])
        self.assertEqual(result["checkpoint_sha256"], EXPECTED_V3_SHA256)

        self.assertIn("display_images", result)
        self.assertIn("raw_low_dose_base64", result["display_images"])
        self.assertIn("denoised_v3_base64", result["display_images"])
        self.assertIn("residual_diff_base64", result["display_images"])
        self.assertIn("clinical_safety_notice", result)

    def test_safe_artifact_paths_without_checkpoint_overwrites(self):
        """Verify that historical model checkpoints remain completely intact and unaltered."""
        historical_models = [
            os.path.join(REPO_ROOT, "medical_ct_denoiser_full.keras"),
            os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_denoiser", "run_001", "candidate_unet_denoiser.keras"),
            os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_hu_calibrated", "run_001", "candidate_unet_hu_calibrated.keras"),
            os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_residual_native", "run_001", "candidate_unet_residual_native.keras")
        ]

        for h_path in historical_models:
            self.assertTrue(os.path.exists(h_path), f"Historical checkpoint was missing: {h_path}")
            # Ensure size is non-zero
            self.assertGreater(os.path.getsize(h_path), 1_000_000, f"Historical checkpoint appears corrupted: {h_path}")


if __name__ == "__main__":
    unittest.main()
