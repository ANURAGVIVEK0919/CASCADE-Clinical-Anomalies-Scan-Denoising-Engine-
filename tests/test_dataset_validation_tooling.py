"""
Unit and regression tests for CASCADE dataset validation tooling:
1. evaluate_data.py
2. validate_models_on_dataset.py
3. validate_dataset_integrity.py

Verifies:
- Empty input directories return safe responses without errors or unexpected disk writes.
- Unpaired input scans suppress denoising reference metrics (PSNR/SSIM gain is None).
- Candidate ROI localization terminology remains non-diagnostic (no clinical diagnosis or severity claims).
- Safe report-output policy (no silent overwrite of existing reports unless overwrite=True is explicit).
- Read-only integrity validation on missing, malformed, or invalid dataset manifests.
- Patient disjointness detects leakage and preserves patient privacy (counts only).
- Matrix dimension and bounding box verification.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate_data import evaluate_directory
from validate_models_on_dataset import run_validation
from validate_dataset_integrity import validate_dataset, save_report


class TestDatasetValidationTooling(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. evaluate_data.py Tests
    # -------------------------------------------------------------------------
    def test_evaluate_directory_empty_folder(self):
        """Verifies evaluate_directory handles an empty directory safely without disk writes or NameError."""
        empty_dir = os.path.join(self.temp_dir.name, "empty_dataset")
        os.makedirs(empty_dir, exist_ok=True)

        # Mock get_diagnostic_model to avoid heavyweight model loading if needed
        mock_model = MagicMock()
        with patch("evaluate_data.get_diagnostic_model", return_value=mock_model):
            res = evaluate_directory(data_dir=empty_dir, max_images=10)

        self.assertIsInstance(res, dict)
        self.assertEqual(res["total_evaluated"], 0)
        self.assertEqual(res["organ_correct"], 0)
        self.assertEqual(res["accuracy"], 0.0)
        self.assertIn("roi_stats", res)
        # Non-diagnostic ROI flags
        for cat in ("high_risk", "medium_risk", "normal"):
            self.assertIn(cat, res["roi_stats"])
            self.assertEqual(res["roi_stats"][cat]["candidate_roi_flagged"], 0)

    def test_evaluate_directory_non_diagnostic_terminology(self):
        """Confirms that candidate ROI counts are labeled as candidate_roi_flagged, not diagnostic conclusions."""
        empty_dir = os.path.join(self.temp_dir.name, "empty_dataset")
        os.makedirs(empty_dir, exist_ok=True)

        mock_model = MagicMock()
        with patch("evaluate_data.get_diagnostic_model", return_value=mock_model):
            res = evaluate_directory(data_dir=empty_dir, max_images=10)

        roi_stats = res["roi_stats"]
        for key, val in roi_stats.items():
            self.assertIn("candidate_roi_flagged", val)
            self.assertNotIn("confirmed_disease", val)
            self.assertNotIn("diagnosis_confirmed", val)

    # -------------------------------------------------------------------------
    # 2. validate_models_on_dataset.py Tests
    # -------------------------------------------------------------------------
    def test_validate_models_empty_data_dir(self):
        """Verifies run_validation handles empty dataset safely without creating output files."""
        empty_dir = os.path.join(self.temp_dir.name, "empty_models_dir")
        os.makedirs(empty_dir, exist_ok=True)

        out_file = os.path.join(self.temp_dir.name, "reports", "test_report.txt")

        mock_diag = MagicMock()
        mock_unet = MagicMock()

        with patch("validate_models_on_dataset.get_diagnostic_model", return_value=mock_diag), \
             patch("validate_models_on_dataset.get_model", return_value=mock_unet):
            res = run_validation(data_dir=empty_dir, sample_size_per_organ=5, output_path=out_file)

        self.assertIsInstance(res, dict)
        self.assertEqual(res["total_images"], 0)
        self.assertEqual(res["paired_images_count"], 0)
        self.assertFalse(res["report_written"])
        self.assertFalse(os.path.exists(out_file), "Empty data must not write report file to disk.")

    def test_validate_models_safe_overwrite_guard(self):
        """Verifies that existing reports are NOT overwritten unless overwrite=True is explicitly passed."""
        out_file = os.path.join(self.temp_dir.name, "existing_report.txt")
        original_content = "ORIGINAL_PRESERVED_CONTENT"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        # Create synthetic scan structure
        scan_dir = os.path.join(self.temp_dir.name, "scans", "chest", "normal", "noisy")
        os.makedirs(scan_dir, exist_ok=True)
        img_path = os.path.join(scan_dir, "slice_001.png")
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img_path)

        mock_diag = MagicMock()
        mock_diag.predict.return_value = (np.array([[1.0, 0.0, 0.0]]), np.array([[0.1, 0.1, 0.9, 0.9]]))
        mock_unet = MagicMock()

        with patch("validate_models_on_dataset.get_diagnostic_model", return_value=mock_diag), \
             patch("validate_models_on_dataset.get_model", return_value=mock_unet), \
             patch("validate_models_on_dataset.process_image", return_value=(
                 np.zeros((256, 256, 1)), np.zeros((256, 256, 1)), np.zeros((256, 256, 1)),
                 {'psnr_noisy': None, 'psnr_restored': None, 'ssim_noisy': None, 'ssim_restored': None, 'improvement': None},
                 None
             )):
            # Case 1: overwrite=False -> Must NOT overwrite
            res1 = run_validation(
                data_dir=os.path.join(self.temp_dir.name, "scans"),
                sample_size_per_organ=1,
                output_path=out_file,
                overwrite=False
            )
            self.assertFalse(res1["report_written"])
            with open(out_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original_content)

            # Case 2: overwrite=True -> Must overwrite
            res2 = run_validation(
                data_dir=os.path.join(self.temp_dir.name, "scans"),
                sample_size_per_organ=1,
                output_path=out_file,
                overwrite=True
            )
            self.assertTrue(res2["report_written"])
            with open(out_file, "r", encoding="utf-8") as f:
                new_content = f.read()
                self.assertIn("CASCADE RESTORATION & DIAGNOSTICS REPORT", new_content)

    def test_validate_models_unpaired_inputs_suppress_metrics(self):
        """Verifies that unpaired scans suppress PSNR/SSIM gains and do not claim denoising improvement."""
        scan_dir = os.path.join(self.temp_dir.name, "scans", "brain", "high_risk", "noisy")
        os.makedirs(scan_dir, exist_ok=True)
        img_path = os.path.join(scan_dir, "slice_unpaired.png")
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img_path)

        out_file = os.path.join(self.temp_dir.name, "unpaired_report.txt")

        mock_diag = MagicMock()
        mock_diag.predict.return_value = (np.array([[0.0, 1.0, 0.0]]), np.array([[0.2, 0.2, 0.8, 0.8]]))
        mock_unet = MagicMock()

        # Simulated unpaired process_image return (metrics with None)
        with patch("validate_models_on_dataset.get_diagnostic_model", return_value=mock_diag), \
             patch("validate_models_on_dataset.get_model", return_value=mock_unet), \
             patch("validate_models_on_dataset.process_image", return_value=(
                 np.zeros((256, 256, 1)), np.zeros((256, 256, 1)), np.zeros((256, 256, 1)),
                 {'psnr_noisy': None, 'psnr_restored': None, 'ssim_noisy': None, 'ssim_restored': None, 'improvement': None},
                 None
             )):
            res = run_validation(
                data_dir=os.path.join(self.temp_dir.name, "scans"),
                sample_size_per_organ=1,
                output_path=out_file
            )

        self.assertEqual(res["total_images"], 1)
        self.assertEqual(res["paired_images_count"], 0)
        self.assertIsNone(res["psnr_gain"])
        self.assertIsNone(res["ssim_gain"])

    # -------------------------------------------------------------------------
    # 3. validate_dataset_integrity.py Tests
    # -------------------------------------------------------------------------
    def test_dataset_integrity_missing_manifest(self):
        """Verifies validate_dataset flags DATA_UNAVAILABLE when manifest is absent."""
        missing_dir = os.path.join(self.temp_dir.name, "non_existent_data")
        res = validate_dataset(processed_dir=missing_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["overall_status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["gates"]["manifest_exists"]["status"], "FAIL")

    def test_dataset_integrity_malformed_json_manifest(self):
        """Verifies validate_dataset flags DATA_INCOMPLETE when manifest JSON is invalid."""
        proc_dir = os.path.join(self.temp_dir.name, "bad_manifest_dir")
        os.makedirs(proc_dir, exist_ok=True)
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            f.write("{ invalid_json: true, ")

        res = validate_dataset(processed_dir=proc_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["overall_status"], "DATA_INCOMPLETE")
        self.assertEqual(res["gates"]["manifest_valid_json"]["status"], "FAIL")

    def test_dataset_integrity_empty_records(self):
        """Verifies validate_dataset flags DATA_INCOMPLETE when records list is empty."""
        proc_dir = os.path.join(self.temp_dir.name, "empty_records_dir")
        os.makedirs(proc_dir, exist_ok=True)
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump({"records": []}, f)

        res = validate_dataset(processed_dir=proc_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["overall_status"], "DATA_INCOMPLETE")
        self.assertEqual(res["gates"]["non_empty"]["status"], "FAIL")

    def test_dataset_integrity_patient_leakage_detection(self):
        """Verifies patient leakage is detected when the same patient index appears in multiple splits."""
        proc_dir = os.path.join(self.temp_dir.name, "leaked_dataset")
        os.makedirs(proc_dir, exist_ok=True)

        # Create valid synthetic images
        img1 = os.path.join(proc_dir, "img1_clean.png")
        img1_n = os.path.join(proc_dir, "img1_noisy.png")
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img1)
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img1_n)

        img2 = os.path.join(proc_dir, "img2_clean.png")
        img2_n = os.path.join(proc_dir, "img2_noisy.png")
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img2)
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img2_n)

        # Patient 123 is present in both train and test splits (leakage)
        manifest = {
            "records": [
                {
                    "image_id": "slice_1",
                    "clean_path": "img1_clean.png",
                    "noisy_path": "img1_noisy.png",
                    "bbox_normalized": [0.1, 0.1, 0.5, 0.5],
                    "coarse_lesion_type": 1,
                    "coarse_lesion_name": "Bone",
                    "patient_index": "Patient_123",
                    "split": "train"
                },
                {
                    "image_id": "slice_2",
                    "clean_path": "img2_clean.png",
                    "noisy_path": "img2_noisy.png",
                    "bbox_normalized": [0.2, 0.2, 0.6, 0.6],
                    "coarse_lesion_type": 1,
                    "coarse_lesion_name": "Bone",
                    "patient_index": "Patient_123",
                    "split": "test"
                }
            ]
        }
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        res = validate_dataset(processed_dir=proc_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["overall_status"], "INTEGRITY_FAILED")
        self.assertEqual(res["gates"]["patient_disjointness"]["status"], "FAIL")
        self.assertTrue(res["gates"]["patient_disjointness"]["patient_leakage_detected"])
        self.assertEqual(res["gates"]["patient_disjointness"]["overlap_counts"]["train_test"], 1)

    def test_dataset_integrity_invalid_bounding_box(self):
        """Verifies that bounding boxes with invalid coordinates fail the bounding_box_validity gate."""
        proc_dir = os.path.join(self.temp_dir.name, "invalid_bbox_dataset")
        os.makedirs(proc_dir, exist_ok=True)

        img1 = os.path.join(proc_dir, "c.png")
        img2 = os.path.join(proc_dir, "n.png")
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img1)
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(img2)

        # ymin > ymax is an invalid bounding box
        manifest = {
            "records": [
                {
                    "image_id": "slice_inv_box",
                    "clean_path": "c.png",
                    "noisy_path": "n.png",
                    "bbox_normalized": [0.8, 0.2, 0.3, 0.6],
                    "coarse_lesion_type": 2,
                    "coarse_lesion_name": "Abdomen",
                    "patient_index": "Patient_A",
                    "split": "train"
                }
            ]
        }
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        res = validate_dataset(processed_dir=proc_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["gates"]["bounding_box_validity"]["status"], "FAIL")
        self.assertEqual(res["gates"]["bounding_box_validity"]["invalid_boxes"], 1)

    def test_dataset_integrity_non_256_matrix_dimension(self):
        """Verifies that non-256x256 image matrices fail decodability/dimension check."""
        proc_dir = os.path.join(self.temp_dir.name, "dimension_mismatch_dataset")
        os.makedirs(proc_dir, exist_ok=True)

        # 128x128 image instead of required 256x256
        img1 = os.path.join(proc_dir, "c.png")
        img2 = os.path.join(proc_dir, "n.png")
        Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(img1)
        Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(img2)

        manifest = {
            "records": [
                {
                    "image_id": "slice_dim",
                    "clean_path": "c.png",
                    "noisy_path": "n.png",
                    "bbox_normalized": [0.1, 0.1, 0.5, 0.5],
                    "coarse_lesion_type": 3,
                    "coarse_lesion_name": "Mediastinum",
                    "patient_index": "Patient_B",
                    "split": "train"
                }
            ]
        }
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        res = validate_dataset(processed_dir=proc_dir, output_report=None)

        self.assertFalse(res["passed"])
        self.assertEqual(res["gates"]["file_decodability"]["status"], "FAIL")
        self.assertGreater(res["gates"]["file_decodability"]["non_256_dimension_images"], 0)

    def test_dataset_integrity_synthetic_clean_pass(self):
        """Verifies that a valid synthetic dataset passes all gates cleanly."""
        proc_dir = os.path.join(self.temp_dir.name, "clean_dataset")
        os.makedirs(proc_dir, exist_ok=True)

        # 3 disjoint patients across train, val, test
        records = []
        for split, pid, ctype, cname in [
            ("train", "P1", 1, "Bone"),
            ("val", "P2", 2, "Abdomen"),
            ("test", "P3", 8, "Pelvis")
        ]:
            c_name = f"{pid}_clean.png"
            n_name = f"{pid}_noisy.png"
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(os.path.join(proc_dir, c_name))
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(os.path.join(proc_dir, n_name))
            records.append({
                "image_id": f"img_{pid}",
                "clean_path": c_name,
                "noisy_path": n_name,
                "bbox_normalized": [0.1, 0.2, 0.4, 0.7],
                "coarse_lesion_type": ctype,
                "coarse_lesion_name": cname,
                "patient_index": pid,
                "split": split
            })

        manifest = {"records": records, "class_distribution": {"Bone": 1, "Abdomen": 1, "Pelvis": 1}}
        with open(os.path.join(proc_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        report_file = os.path.join(self.temp_dir.name, "report.json")
        res = validate_dataset(
            processed_dir=proc_dir,
            legacy_brain_dir=os.path.join(self.temp_dir.name, "no_legacy"),
            output_report=report_file,
            overwrite=True
        )

        self.assertTrue(res["passed"])
        self.assertEqual(res["overall_status"], "PASSED_FOR_LOCALIZATION_AND_DENOISING")
        self.assertTrue(res["report_written"])
        self.assertTrue(os.path.exists(report_file))


if __name__ == "__main__":
    unittest.main()
