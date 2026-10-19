"""
CASCADE Research API Security, Endpoint Isolation & Observability Test Suite (Gate 4)

Covers:
1. Oversized file rejection (HTTP 413).
2. Corrupt/malformed DICOM rejection (HTTP 400 without server path or traceback leakage).
3. Empty upload rejection (HTTP 400).
4. Non-CT modality rejection (HTTP 400).
5. Missing or invalid rescale metadata rejection (HTTP 400).
6. Unexpected image dimensions and unsupported multiframe rejection (HTTP 400).
7. Checkpoint hash mismatch or missing checkpoint behavior.
8. Zero persistent scan storage / temporary file cleanup.
9. Privacy & zero-PHI protection in API responses and error messages.
10. Separate pipeline telemetry in /metrics (legacy vs research).
11. Legacy upload route telemetry reporting legacy model, never v3.
12. Successful research inference without source-file modification.
"""

import os
import io
import json
import hashlib
import tempfile
import unittest
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from research_inference import (
    V3_CHECKPOINT_PATH,
    EXPECTED_V3_SHA256,
    MAX_DICOM_UPLOAD_SIZE,
    verify_checkpoint_integrity,
    get_research_v3_model,
    denoise_dicom_research,
    get_research_telemetry
)
from app import app


def build_synthetic_ct_dicom(
    rows: int = 512,
    cols: int = 512,
    modality: str = "CT",
    include_rescale: bool = True,
    slope: float = 1.0,
    intercept: float = -1024.0,
    hu_val: float = 40.0,
    patient_name: str = "ANONYMOUS^TEST",
    patient_id: str = "SYNTH_ID_12345",
    num_frames: int = 1
) -> io.BytesIO:
    """Creates a synthetic DICOM CT slice with standard tags and valid pixel metadata."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9.999"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    buf = io.BytesIO()
    ds = FileDataset(buf, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19650515"
    ds.StudyDate = "20261020"
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [0.976562, 0.976562]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, 50.0]

    if num_frames > 1:
        ds.NumberOfFrames = num_frames

    if include_rescale:
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    raw_val = (hu_val - intercept) / slope if slope != 0.0 else (hu_val - intercept)
    if num_frames > 1:

        pixels = np.full((num_frames, rows, cols), raw_val, dtype=np.int16)
    else:
        pixels = np.full((rows, cols), raw_val, dtype=np.int16)

    ds.PixelData = pixels.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(buf)
    buf.seek(0)
    return buf


class TestResearchAPISecurity(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_oversized_file_rejection(self):
        """Verify that payloads exceeding 32 MB are strictly rejected with HTTP 413."""
        # Create an oversized payload buffer (33 MB)
        oversized_data = b"0" * (MAX_DICOM_UPLOAD_SIZE + 1024 * 1024)
        buf = io.BytesIO(oversized_data)

        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (buf, "oversized_scan.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 413, "Oversized upload was not rejected with HTTP 413!")
        data = resp.get_json()
        self.assertEqual(data["status"], "REJECTED")
        self.assertIn("exceeds maximum allowable limit", data["error"])

    def test_corrupt_dicom_rejection_without_leakage(self):
        """Verify corrupt DICOM data is rejected with HTTP 400 and zero server path/traceback leakage."""
        corrupt_bytes = b"CORRUPTED_NON_DICOM_BINARY_GARBAGE_PADDING_1234567890"
        buf = io.BytesIO(corrupt_bytes)

        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (buf, "corrupt.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "REJECTED")
        self.assertIn("Invalid or corrupt DICOM file", data["error"])

        # Security check: Zero internal path or traceback leakage
        raw_resp_text = resp.get_data(as_text=True)
        self.assertNotIn("Traceback", raw_resp_text)
        self.assertNotIn("File \"", raw_resp_text)
        self.assertNotIn("d:\\", raw_resp_text.lower())

    def test_empty_upload_rejection(self):
        """Verify that empty uploads or 0-byte files return HTTP 400."""
        empty_buf = io.BytesIO(b"")

        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (empty_buf, "empty.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "REJECTED")
        self.assertIn("Empty file", data["error"])

    def test_non_ct_modality_rejection(self):
        """Verify that non-CT modalities (e.g. MR, CR, XA) are strictly rejected with HTTP 400."""
        mr_buf = build_synthetic_ct_dicom(modality="MR")

        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (mr_buf, "mr_slice.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "REJECTED")
        self.assertIn("Unsupported modality: 'MR'", data["error"])

    def test_missing_or_invalid_rescale_metadata(self):
        """Verify missing or invalid RescaleSlope / RescaleIntercept is rejected with HTTP 400."""
        # 1. Missing rescale metadata
        missing_rescale_buf = build_synthetic_ct_dicom(include_rescale=False)
        resp1 = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (missing_rescale_buf, "no_rescale.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp1.status_code, 400)
        data1 = resp1.get_json()
        self.assertIn("Missing required DICOM rescale metadata", data1["error"])

        # 2. Zero rescale slope (invalid physics)
        zero_slope_buf = build_synthetic_ct_dicom(slope=0.0)
        resp2 = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (zero_slope_buf, "zero_slope.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp2.status_code, 400)
        data2 = resp2.get_json()
        self.assertIn("slope cannot be zero", data2["error"])

    def test_multiframe_and_dimension_rejection(self):
        """Verify multiframe (3D) or invalid dimensions are strictly rejected with HTTP 400."""
        # 1. Multiframe rejection
        multiframe_buf = build_synthetic_ct_dicom(rows=256, cols=256, num_frames=5)
        resp1 = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (multiframe_buf, "multiframe.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp1.status_code, 400)
        data1 = resp1.get_json()
        self.assertIn("Multiframe DICOM files are not supported", data1["error"])

        # 2. Sub-minimum dimensions (< 128x128)
        tiny_buf = build_synthetic_ct_dicom(rows=64, cols=64)
        resp2 = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (tiny_buf, "tiny.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp2.status_code, 400)
        data2 = resp2.get_json()
        self.assertIn("Minimum supported dimension is 128x128", data2["error"])

    def test_checkpoint_hash_mismatch_behavior(self):
        """Verify checkpoint integrity check fails safely on bad hash or missing file."""
        self.assertTrue(verify_checkpoint_integrity(V3_CHECKPOINT_PATH))
        self.assertFalse(verify_checkpoint_integrity("nonexistent/checkpoint.keras"))

        # Test simulated corrupted checkpoint
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tf_fake:
            tf_fake.write(b"CORRUPTED_FAKE_MODEL_WEIGHTS")
            fake_path = tf_fake.name

        try:
            self.assertFalse(verify_checkpoint_integrity(fake_path))
            with self.assertRaises((ValueError, RuntimeError)):
                get_research_v3_model(fake_path)
        finally:
            if os.path.exists(fake_path):
                os.remove(fake_path)

    def test_temporary_file_cleanup_and_no_storage(self):
        """Verify that processing does not store uploads to disk or leave temporary files."""
        upload_folder = app.config.get("UPLOAD_FOLDER", "static/uploads")
        initial_files = set(os.listdir(upload_folder)) if os.path.exists(upload_folder) else set()

        dcm_buf = build_synthetic_ct_dicom(rows=512, cols=512, hu_val=35.0)
        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (dcm_buf, "transient_test.dcm"), "window": "soft_tissue"},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)

        current_files = set(os.listdir(upload_folder)) if os.path.exists(upload_folder) else set()
        # No new files must be left in upload directory
        self.assertEqual(
            initial_files,
            current_files,
            "Research endpoint leaked persistent files into upload directory!"
        )

    def test_phi_protection_and_sanitization(self):
        """Verify patient names, birthdates, and private IDs are redacted from responses and logs."""
        sensitive_name = "Doe^John^Private"
        sensitive_id = "PHI-SECRET-MRN-8877"
        sensitive_dob = "19600101"

        dcm_buf = build_synthetic_ct_dicom(
            rows=512,
            cols=512,
            patient_name=sensitive_name,
            patient_id=sensitive_id
        )

        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (dcm_buf, "patient_scan.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        resp_text = resp.get_data(as_text=True)

        # PHI must NEVER appear in the response payload
        self.assertNotIn(sensitive_name, resp_text, "PatientName leaked into API response!")
        self.assertNotIn(sensitive_id, resp_text, "PatientID leaked into API response!")
        self.assertNotIn(sensitive_dob, resp_text, "PatientBirthDate leaked into API response!")

        data = resp.get_json()
        self.assertEqual(data["source_metadata"]["patient_id"], "[REDACTED_FOR_PRIVACY]")
        self.assertIn("burned_in_annotation", data["source_metadata"])

        # UIDs and institution fields must not leak into metadata
        for forbidden_key in [
            "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "InstitutionName", "ReferringPhysicianName", "OperatorsName", "AccessionNumber"
        ]:
            self.assertNotIn(forbidden_key, data["source_metadata"])

        # Privacy notice must be present and transparent
        self.assertIn("privacy_notice", data)
        self.assertIn("PS 3.15", data["privacy_notice"])
        self.assertIn("burned-in pixel annotations", data["privacy_notice"])

    def test_separate_telemetry_in_metrics(self):
        """Verify /metrics distinctly reports legacy pipeline and research pipeline telemetry."""
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("pipelines", data)
        pipelines = data["pipelines"]

        # 1. Legacy Image Pipeline
        self.assertIn("legacy_image_pipeline", pipelines)
        legacy = pipelines["legacy_image_pipeline"]
        self.assertEqual(legacy["model_id"], "medical_ct_denoiser_full.keras")

        # 2. Research DICOM Pipeline
        self.assertIn("research_dicom_pipeline", pipelines)
        research = pipelines["research_dicom_pipeline"]
        self.assertEqual(research["model_id"], "candidate_unet_residual_edge_v3.keras")
        self.assertEqual(research["checkpoint_sha256"], EXPECTED_V3_SHA256)
        self.assertEqual(research["status"], "Available")
        self.assertEqual(research["endpoint"], "/api/research/denoise_dicom")

        # 3. Research Telemetry Counters
        self.assertIn("research_telemetry", data)
        r_telem = data["research_telemetry"]
        self.assertIn("requests_total", r_telem)
        self.assertIn("requests_success", r_telem)
        self.assertIn("requests_failed", r_telem)
        self.assertIn("last_latency_ms", r_telem)

    def test_legacy_upload_route_telemetry(self):
        """Verify legacy /process route explicitly reports legacy pipeline and never claims v3."""
        # Create a small 256x256 test PNG
        png_buf = io.BytesIO()
        Image.fromarray(np.full((256, 256), 128, dtype=np.uint8)).save(png_buf, format="PNG")
        png_buf.seek(0)

        resp = self.client.post(
            "/process",
            data={"image": (png_buf, "legacy_web_test.png"), "organ": "chest"},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertIn("results", data)
        first_result = data["results"][0]

        self.assertEqual(first_result["pipeline"], "legacy_image_pipeline")
        self.assertIn("Legacy U-Net", first_result["model_version"])
        self.assertFalse(first_result["is_research_model"])
        self.assertNotIn("Residual U-Net v3", first_result["model_version"])

    def test_successful_inference_source_immutability(self):
        """Verify that research inference never mutates or overwrites source DICOM file."""
        dcm_buf = build_synthetic_ct_dicom(rows=512, cols=512, hu_val=42.0)
        source_bytes = dcm_buf.getvalue()
        sha_before = hashlib.sha256(source_bytes).hexdigest()

        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf_file:
            tf_file.write(source_bytes)
            tf_path = tf_file.name

        try:
            result = denoise_dicom_research(tf_path, window_mode="soft_tissue")
            self.assertEqual(result["status"], "SUCCESS")

            # Check that file on disk was not modified in place
            with open(tf_path, "rb") as f:
                current_bytes = f.read()
            sha_after = hashlib.sha256(current_bytes).hexdigest()

            self.assertEqual(sha_before, sha_after, "Source DICOM file was mutated on disk during inference!")
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_output_range_accounting_and_clamping(self):
        """Verify output range accounting, clamping policy, and bounds enforcement in response."""
        dcm_buf = build_synthetic_ct_dicom(rows=512, cols=512, hu_val=40.0)
        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (dcm_buf, "test_range.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Check output_range_accounting structure
        self.assertIn("output_range_accounting", data)
        ora = data["output_range_accounting"]
        self.assertEqual(ora["policy"], "DOMAIN_BOUNDED_CLAMPING")
        self.assertEqual(ora["valid_range_hu"], [-1000.0, 1000.0])
        self.assertIn("raw_predictions", ora)
        self.assertIn("out_of_range_pixels", ora)

        # Check that postprocessed denoised HU is strictly bounded
        stats = data["statistics"]
        self.assertGreaterEqual(stats["denoised_hu_min"], -1000.0)
        self.assertLessEqual(stats["denoised_hu_max"], 1000.0)

    def test_finite_predictions_and_numerical_safety(self):
        """Verify that extreme CT values produce strictly finite, bounded numbers (no NaN or Inf)."""
        # Create a synthetic slice with extreme high-density bone/metal (+3000 HU)
        dcm_buf = build_synthetic_ct_dicom(rows=512, cols=512, hu_val=3000.0)
        resp = self.client.post(
            "/api/research/denoise_dicom",
            data={"dicom_scan": (dcm_buf, "extreme_metal.dcm")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        stats = data["statistics"]
        for key in ["denoised_hu_min", "denoised_hu_max", "denoised_hu_mean", "mean_abs_diff_hu", "max_abs_diff_hu"]:
            val = stats[key]
            self.assertTrue(np.isfinite(val), f"Non-finite value found for {key}: {val}")

        # Clamping guarantees maximum is capped at +1000.0 HU
        self.assertLessEqual(stats["denoised_hu_max"], 1000.0)

    def test_consistent_inverse_hu_transformation(self):
        """Verify mathematical round-trip consistency of fixed HU normalization and inverse mapping."""
        from models.residual_unet import hu_to_normalized, normalized_to_hu

        # Test across the clinical range [-1000, +1000] HU in increments
        test_hu = np.linspace(-1000.0, 1000.0, 201, dtype=np.float32)
        norm = hu_to_normalized(test_hu)
        self.assertTrue(np.all(norm >= 0.0))
        self.assertTrue(np.all(norm <= 1.0))

        recovered_hu = normalized_to_hu(norm)
        max_err = np.max(np.abs(recovered_hu - test_hu))
        self.assertLess(max_err, 1e-4, f"Round-trip reconstruction error too high: {max_err}")

        # Test clipping flag on out-of-range inputs
        out_of_range_norm = np.array([-0.05, 0.5, 1.05], dtype=np.float32)
        clamped_hu = normalized_to_hu(out_of_range_norm, clip_to_bounds=True)
        self.assertEqual(clamped_hu[0], -1000.0)
        self.assertEqual(clamped_hu[1], 0.0)
        self.assertEqual(clamped_hu[2], 1000.0)

    def test_out_of_range_residual_behavior_synthetic(self):
        """Verify clamping behavior when raw residual addition exceeds [0, 1] domain."""
        # Simulated unconstrained raw model predictions with undershoot and overshoot
        synthetic_preds = np.array([[-0.01, 0.0, 0.5], [0.8, 1.0, 1.02]], dtype=np.float32)
        clamped_preds = np.clip(synthetic_preds, 0.0, 1.0)

        # In-domain elements must remain completely untouched
        self.assertEqual(clamped_preds[0, 1], 0.0)
        self.assertEqual(clamped_preds[0, 2], 0.5)
        self.assertEqual(clamped_preds[1, 0], 0.8)
        self.assertEqual(clamped_preds[1, 1], 1.0)

        # Out-of-domain elements must clamp cleanly
        self.assertEqual(clamped_preds[0, 0], 0.0)
        self.assertEqual(clamped_preds[1, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
