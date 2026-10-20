"""
Unit & Regression Tests for Research Model Reproducibility & Benchmark Tooling:
- train_residual_unet_v3.py
- evaluate_residual_unet_v3.py

Verifies:
1. Training dry-run by default without training execution or checkpoint creation.
2. Training patient-disjoint isolation verification & leakage detection.
3. Checkpoint anti-overwrite safety guard when overwrite=False.
4. Evaluation safe missing data / manifest handling (clean DATA_UNAVAILABLE status).
5. Evaluation report overwrite safeguards (prevents silent overwrite of historical reports).
6. Exact numerical calculation of consistent soft-tissue PSNR (400 HU range) vs historical mixed-domain PSNR (2000 HU range).
7. Metric domain documentation semantic integrity (explicit soft-tissue and full HU range definitions).
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_residual_unet_v3 import run_training, validate_isolation
from evaluate_residual_unet_v3 import (
    run_evaluation,
    save_evaluation_report,
    compute_slice_metrics,
    aggregate_cohort_metrics,
    aggregate_metric
)


class TestResearchReproducibilityTooling(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_manifests(self):
        """Creates temporary synthetic manifests for testing."""
        dev_path = os.path.join(self.temp_dir.name, "dev_manifest.json")
        v2_path = os.path.join(self.temp_dir.name, "v2_manifest.json")
        v3_path = os.path.join(self.temp_dir.name, "v3_manifest.json")

        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump({"train_cohort": ["P1", "P2"], "val_cohort": ["P3"]}, f)
        with open(v2_path, "w", encoding="utf-8") as f:
            json.dump({"cohort": ["P4", "P5"]}, f)
        with open(v3_path, "w", encoding="utf-8") as f:
            json.dump({"cohort": ["P6", "P7"]}, f)

        return dev_path, v2_path, v3_path

    # -------------------------------------------------------------------------
    # 1. train_residual_unet_v3.py Tests
    # -------------------------------------------------------------------------
    def test_train_dry_run_by_default(self):
        """Verifies training runs in dry-run mode when execute=False without disk side-effects."""
        dev_m, v2_m, v3_m = self._create_mock_manifests()
        out_dir = os.path.join(self.temp_dir.name, "train_out")
        target_ckpt = os.path.join(out_dir, "test_candidate.keras")

        res = run_training(
            data_dir=self.temp_dir.name,
            dev_manifest=dev_m,
            v2_manifest=v2_m,
            v3_manifest=v3_m,
            output_dir=out_dir,
            checkpoint_path=target_ckpt,
            execute=False,
            overwrite=False
        )

        self.assertEqual(res["status"], "DRY_RUN_PASSED")
        self.assertTrue(res["dry_run"])
        self.assertTrue(res["isolation_verified"])
        self.assertEqual(res["train_cohort_count"], 2)
        self.assertEqual(res["val_cohort_count"], 1)
        self.assertFalse(os.path.exists(target_ckpt), "Dry-run must not create checkpoint file.")

    def test_train_isolation_leakage_detection(self):
        """Verifies that patient leakage between train cohort and locked test cohorts triggers an assertion."""
        dev_path = os.path.join(self.temp_dir.name, "leaked_dev.json")
        v2_path = os.path.join(self.temp_dir.name, "v2.json")
        v3_path = os.path.join(self.temp_dir.name, "v3.json")

        # P_LEAK is in both train and locked v3
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump({"train_cohort": ["P1", "P_LEAK"], "val_cohort": ["P2"]}, f)
        with open(v2_path, "w", encoding="utf-8") as f:
            json.dump({"cohort": ["P3"]}, f)
        with open(v3_path, "w", encoding="utf-8") as f:
            json.dump({"cohort": ["P_LEAK", "P4"]}, f)

        with self.assertRaises(AssertionError) as ctx:
            validate_isolation(dev_path, v2_path, v3_path)
        self.assertIn("LEAKAGE", str(ctx.exception))

    def test_train_safe_overwrite_guard(self):
        """Verifies training aborts before executing if checkpoint exists and overwrite=False."""
        dev_m, v2_m, v3_m = self._create_mock_manifests()
        out_dir = os.path.join(self.temp_dir.name, "train_out")
        os.makedirs(out_dir, exist_ok=True)
        target_ckpt = os.path.join(out_dir, "existing_candidate.keras")

        # Create dummy checkpoint
        with open(target_ckpt, "w", encoding="utf-8") as f:
            f.write("ORIGINAL_FROZEN_WEIGHTS")

        res = run_training(
            data_dir=self.temp_dir.name,
            dev_manifest=dev_m,
            v2_manifest=v2_m,
            v3_manifest=v3_m,
            output_dir=out_dir,
            checkpoint_path=target_ckpt,
            execute=True,
            overwrite=False
        )

        self.assertEqual(res["status"], "CHECKPOINT_EXISTS_ABORTED")
        self.assertEqual(res["error"], "CHECKPOINT_EXISTS")
        with open(target_ckpt, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "ORIGINAL_FROZEN_WEIGHTS", "Existing checkpoint must not be modified.")

    # -------------------------------------------------------------------------
    # 2. evaluate_residual_unet_v3.py Tests
    # -------------------------------------------------------------------------
    def test_evaluate_missing_manifest_handling(self):
        """Verifies evaluation reports DATA_UNAVAILABLE when manifest is missing without crashing."""
        missing_m = os.path.join(self.temp_dir.name, "missing_manifest.json")
        res = run_evaluation(manifest_path=missing_m)

        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["error"], "MANIFEST_NOT_FOUND")

    def test_evaluate_missing_dicom_pairs_handling(self):
        """Verifies evaluation reports DATA_UNAVAILABLE when raw DICOM pairs are missing without fabricating metrics."""
        m_path = os.path.join(self.temp_dir.name, "test_manifest.json")
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump({"cohort": ["L999"]}, f)

        # Mock existing candidate checkpoint
        dummy_ckpt = os.path.join(self.temp_dir.name, "dummy_model.keras")
        with open(dummy_ckpt, "w", encoding="utf-8") as f:
            f.write("DUMMY_CKPT")

        res = run_evaluation(
            data_dir=os.path.join(self.temp_dir.name, "empty_scans"),
            manifest_path=m_path,
            v3_checkpoint=dummy_ckpt
        )

        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["error"], "DICOM_PAIRS_NOT_FOUND")
        self.assertFalse(res["report_written"])

    def test_evaluate_safe_overwrite_guard(self):
        """Verifies that evaluation reports cannot be silently overwritten unless overwrite=True is explicit."""
        out_report = os.path.join(self.temp_dir.name, "existing_report.json")
        original_data = {"test_run": "HISTORICAL_REPORT_V1"}
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(original_data, f)

        new_data = {"test_run": "OVERWRITE_ATTEMPT_V2"}

        # Attempt 1: overwrite=False -> Must NOT overwrite
        written1 = save_evaluation_report(new_data, out_report, overwrite=False)
        self.assertFalse(written1)
        with open(out_report, "r", encoding="utf-8") as f:
            content1 = json.load(f)
            self.assertEqual(content1["test_run"], "HISTORICAL_REPORT_V1")

        # Attempt 2: overwrite=True -> Must overwrite
        written2 = save_evaluation_report(new_data, out_report, overwrite=True)
        self.assertTrue(written2)
        with open(out_report, "r", encoding="utf-8") as f:
            content2 = json.load(f)
            self.assertEqual(content2["test_run"], "OVERWRITE_ATTEMPT_V2")

    def test_numerical_psnr_and_error_domain_distinction(self):
        """
        Numerically verifies the exact mathematical formula for:
        - Consistent soft-tissue PSNR (MAX_I = 400 HU)
        - Historical mixed-domain PSNR (MAX_I = 2000 HU)
        - The exact +13.98 dB offset = 20 * log10(2000 / 400)
        Using a controlled synthetic slice array.
        """
        # Create 64x64 synthetic slice with air background (-1000 HU)
        fd_hu = np.full((64, 64), -1000.0, dtype=np.float32)
        # Soft-tissue region: 40 HU (typical muscle/liver)
        fd_hu[16:48, 16:48] = 40.0
        # Bone region: 800 HU
        fd_hu[0:16, 0:16] = 800.0

        # Construct prediction with constant error of exactly +4.0 HU in soft-tissue
        p_hu = fd_hu.copy()
        p_hu[16:48, 16:48] += 4.0
        # High error in bone to verify soft-tissue masking excludes bone
        p_hu[0:16, 0:16] += 50.0

        st_mask = (fd_hu >= -160.0) & (fd_hu <= 240.0)
        body_mask = fd_hu > -500.0

        p_norm = (np.clip(p_hu, -1000.0, 1000.0) + 1000.0) / 2000.0
        fd_norm = (np.clip(fd_hu, -1000.0, 1000.0) + 1000.0) / 2000.0

        metrics = compute_slice_metrics(p_hu, fd_hu, p_norm, fd_norm, st_mask, body_mask)

        # 1. Soft-tissue MAE must be exactly 4.0 HU (ignoring bone error of 50 HU)
        self.assertAlmostEqual(metrics["st_mae"], 4.0, places=4)
        self.assertAlmostEqual(metrics["st_rmse"], 4.0, places=4)

        # 2. Consistent soft-tissue PSNR: 20 * log10(400.0 / 4.0) = 20 * log10(100) = 40.00 dB
        expected_st_psnr = 20.0 * np.log10(400.0 / 4.0)
        self.assertAlmostEqual(expected_st_psnr, 40.0, places=4)
        self.assertAlmostEqual(metrics["consistent_st_psnr"], 40.0, places=4)

        # 3. Historical mixed-domain PSNR: 20 * log10(2000.0 / 4.0) = 20 * log10(500) = 53.9794 dB
        expected_mixed_psnr = 20.0 * np.log10(2000.0 / 4.0)
        self.assertAlmostEqual(expected_mixed_psnr, 53.9794, places=3)
        self.assertAlmostEqual(metrics["historical_mixed_psnr"], expected_mixed_psnr, places=3)

        # 4. Exact mathematical offset: 20 * log10(5) = 13.9794 dB
        expected_offset = 20.0 * np.log10(2000.0 / 400.0)
        observed_offset = metrics["historical_mixed_psnr"] - metrics["consistent_st_psnr"]
        self.assertAlmostEqual(observed_offset, expected_offset, places=4)
        self.assertAlmostEqual(observed_offset, 13.9794, places=3)

    def test_evaluate_metric_documentation_integrity(self):
        """Verifies report documentation explicitly details data ranges, masks, and mathematical formulas."""
        out_report = os.path.join(self.temp_dir.name, "doc_report.json")
        sample_report = {
            "metric_documentation": {
                "consistent_st_psnr_db": {
                    "definition": "20 * log10(400.0 / (st_rmse + 1e-8))",
                    "numerator_range_hu": 400.0,
                    "window": "[-160, +240] HU",
                    "mask": "Soft-tissue mask (ground-truth reference in [-160, 240] HU)",
                    "description": "Consistent physical soft-tissue PSNR"
                },
                "historical_mixed_psnr_db": {
                    "definition": "20 * log10(2000.0 / (st_rmse + 1e-8))",
                    "numerator_range_hu": 2000.0,
                    "offset_relative_to_consistent_st_psnr": "+13.98 dB (20 * log10(5))",
                    "description": "Historical mixed-domain PSNR"
                },
                "st_mae_hu": {
                    "definition": "mean(abs(pred_hu - ref_hu)) over st_mask",
                    "units": "Hounsfield Units (HU)",
                    "window": "[-160, +240] HU"
                },
                "ssim": {
                    "definition": "tf.image.ssim(pred_norm, ref_norm, max_val=1.0)",
                    "inputs": "Full 512x512 unmasked slices normalized to [0, 1] via [-1000, +1000] HU",
                    "mask": "None (unmasked full slice)"
                },
                "vector_gradient_error": {
                    "definition": "mean(sqrt((gx_pred - gx_ref)^2 + (gy_pred - gy_ref)^2)) on edge_mask",
                    "units": "HU/pixel",
                    "operator": "2D Sobel 3x3 kernels (gx, gy)"
                }
            }
        }
        written = save_evaluation_report(sample_report, out_report, overwrite=True)
        self.assertTrue(written)
        with open(out_report, "r", encoding="utf-8") as f:
            doc = json.load(f)["metric_documentation"]
            self.assertEqual(doc["consistent_st_psnr_db"]["numerator_range_hu"], 400.0)
            self.assertEqual(doc["historical_mixed_psnr_db"]["numerator_range_hu"], 2000.0)
            self.assertIn("+13.98 dB", doc["historical_mixed_psnr_db"]["offset_relative_to_consistent_st_psnr"])
            self.assertEqual(doc["st_mae_hu"]["units"], "Hounsfield Units (HU)")
            self.assertIn("max_val=1.0", doc["ssim"]["definition"])
            self.assertEqual(doc["vector_gradient_error"]["units"], "HU/pixel")

    def test_compute_slice_metrics_empty_soft_tissue_mask(self):
        """Verifies compute_slice_metrics returns None (null) for ST metrics when soft-tissue mask is empty."""
        slice_a = np.full((32, 32), 40.0, dtype=np.float32)
        norm_a = np.full((32, 32), 0.52, dtype=np.float32)
        empty_st = np.zeros((32, 32), dtype=bool)
        valid_body = np.ones((32, 32), dtype=bool)

        metrics = compute_slice_metrics(slice_a, slice_a, norm_a, norm_a, empty_st, valid_body)
        self.assertIsNone(metrics["st_mae"])
        self.assertIsNone(metrics["st_rmse"])
        self.assertIsNone(metrics["consistent_st_psnr"])
        self.assertIsNone(metrics["historical_mixed_psnr"])
        self.assertIsNone(metrics["vector_gradient_error"])
        self.assertIsNone(metrics["st_vector_gradient_error"])
        self.assertIsNone(metrics["scalar_edge_error"])
        # Body metrics and SSIM remain valid
        self.assertIsNotNone(metrics["global_mae"])
        self.assertIsNotNone(metrics["global_rmse"])
        self.assertIsNotNone(metrics["global_body_psnr"])
        self.assertIsNotNone(metrics["ssim"])

    def test_compute_slice_metrics_empty_body_mask(self):
        """Verifies compute_slice_metrics returns None (null) for body metrics when body mask is empty."""
        slice_a = np.full((32, 32), 40.0, dtype=np.float32)
        norm_a = np.full((32, 32), 0.52, dtype=np.float32)
        valid_st = np.ones((32, 32), dtype=bool)
        empty_body = np.zeros((32, 32), dtype=bool)

        metrics = compute_slice_metrics(slice_a, slice_a, norm_a, norm_a, valid_st, empty_body)
        self.assertIsNone(metrics["global_mae"])
        self.assertIsNone(metrics["global_rmse"])
        self.assertIsNone(metrics["global_body_psnr"])
        # ST metrics and SSIM remain valid
        self.assertIsNotNone(metrics["st_mae"])
        self.assertIsNotNone(metrics["st_rmse"])
        self.assertIsNotNone(metrics["consistent_st_psnr"])
        self.assertIsNotNone(metrics["ssim"])

    def test_compute_slice_metrics_empty_both_masks(self):
        """Verifies compute_slice_metrics never returns fabricated 0.0 error or PSNR when masks are empty."""
        slice_a = np.zeros((32, 32), dtype=np.float32)
        norm_a = np.zeros((32, 32), dtype=np.float32)
        empty_mask = np.zeros((32, 32), dtype=bool)

        metrics = compute_slice_metrics(slice_a, slice_a, norm_a, norm_a, empty_mask, empty_mask)
        self.assertIsNone(metrics["st_mae"])
        self.assertIsNone(metrics["st_rmse"])
        self.assertIsNone(metrics["consistent_st_psnr"])
        self.assertIsNone(metrics["historical_mixed_psnr"])
        self.assertIsNone(metrics["global_body_psnr"])
        self.assertIsNone(metrics["vector_gradient_error"])
        # Full unmasked normalized slice SSIM remains measurable
        self.assertIsNotNone(metrics["ssim"])

    def test_aggregation_mixed_valid_and_empty_mask_slices(self):
        """Verifies that empty-mask slices (None) do not dilute or bias arithmetic means of valid slices."""
        metrics_by_system = {
            "raw_low_dose": [
                {
                    "st_mae": 10.0, "st_rmse": 10.0, "consistent_st_psnr": 32.04,
                    "historical_mixed_psnr": 46.02, "global_body_psnr": 40.0,
                    "ssim": 0.91, "vector_gradient_error": 18.0,
                    "st_vector_gradient_error": 18.0, "scalar_edge_error": 12.0
                },
                {
                    "st_mae": None, "st_rmse": None, "consistent_st_psnr": None,
                    "historical_mixed_psnr": None, "global_body_psnr": None,
                    "ssim": 0.89, "vector_gradient_error": None,
                    "st_vector_gradient_error": None, "scalar_edge_error": None
                }
            ],
            "residual_unet_v3": [
                {
                    "st_mae": 8.0, "st_rmse": 8.0, "consistent_st_psnr": 33.98,
                    "historical_mixed_psnr": 47.96, "global_body_psnr": 42.0,
                    "ssim": 0.94, "vector_gradient_error": 15.0,
                    "st_vector_gradient_error": 15.0, "scalar_edge_error": 10.0
                },
                {
                    "st_mae": None, "st_rmse": None, "consistent_st_psnr": None,
                    "historical_mixed_psnr": None, "global_body_psnr": None,
                    "ssim": 0.92, "vector_gradient_error": None,
                    "st_vector_gradient_error": None, "scalar_edge_error": None
                }
            ]
        }

        summary, audit = aggregate_cohort_metrics(metrics_by_system, ["raw_low_dose", "residual_unet_v3"])

        # ST-MAE must be exactly 8.00 (from valid slice 1), NOT diluted to 4.00
        self.assertEqual(summary["residual_unet_v3"]["st_mae_hu"], 8.00)
        self.assertEqual(summary["raw_low_dose"]["st_mae_hu"], 10.00)
        self.assertEqual(summary["residual_unet_v3"]["valid_slice_counts"]["soft_tissue"], 1)
        self.assertEqual(summary["residual_unet_v3"]["valid_slice_counts"]["total_evaluated"], 2)

        # SSIM is evaluated across both slices: (0.94 + 0.92) / 2 = 0.9300
        self.assertEqual(summary["residual_unet_v3"]["ssim"], 0.9300)
        self.assertEqual(summary["residual_unet_v3"]["valid_slice_counts"]["ssim"], 2)

        # Acceptance audit uses valid-only differences: 8.00 - 10.00 = -2.00 HU
        self.assertEqual(audit["primary_criterion_mae"]["delta_hu"], -2.00)
        self.assertEqual(audit["overall_status"], "PASS")

    def test_aggregation_all_slices_empty_mask_insufficient_data(self):
        """Verifies that when all cohort slices have empty masks, aggregate returns None and INSUFFICIENT_DATA."""
        metrics_by_system = {
            "raw_low_dose": [
                {
                    "st_mae": None, "st_rmse": None, "consistent_st_psnr": None,
                    "historical_mixed_psnr": None, "global_body_psnr": None,
                    "ssim": 0.90, "vector_gradient_error": None,
                    "st_vector_gradient_error": None, "scalar_edge_error": None
                }
            ],
            "residual_unet_v3": [
                {
                    "st_mae": None, "st_rmse": None, "consistent_st_psnr": None,
                    "historical_mixed_psnr": None, "global_body_psnr": None,
                    "ssim": 0.92, "vector_gradient_error": None,
                    "st_vector_gradient_error": None, "scalar_edge_error": None
                }
            ]
        }

        summary, audit = aggregate_cohort_metrics(metrics_by_system, ["raw_low_dose", "residual_unet_v3"])

        self.assertIsNone(summary["residual_unet_v3"]["st_mae_hu"])
        self.assertEqual(summary["residual_unet_v3"]["valid_slice_counts"]["soft_tissue"], 0)
        self.assertEqual(audit["overall_status"], "INSUFFICIENT_DATA")
        self.assertEqual(audit["primary_criterion_mae"]["status"], "UNAVAILABLE")
        self.assertFalse(audit["primary_criterion_mae"]["passed"])

    def test_report_serialization_of_null_metrics(self):
        """Verifies evaluation reports with None (null) metrics serialize and parse cleanly."""
        out_report = os.path.join(self.temp_dir.name, "null_metrics_report.json")
        sample_data = {
            "summary": {
                "st_mae_hu": None,
                "consistent_st_psnr_db": None,
                "ssim": 0.9421
            },
            "valid_slice_counts": {
                "soft_tissue": 0,
                "ssim": 10
            }
        }

        saved = save_evaluation_report(sample_data, out_report, overwrite=True)
        self.assertTrue(saved)

        with open(out_report, "r", encoding="utf-8") as f:
            raw_text = f.read()
            self.assertIn('"st_mae_hu": null', raw_text)
            self.assertIn('"consistent_st_psnr_db": null', raw_text)

        with open(out_report, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            self.assertIsNone(parsed["summary"]["st_mae_hu"])
            self.assertIsNone(parsed["summary"]["consistent_st_psnr_db"])
            self.assertEqual(parsed["summary"]["ssim"], 0.9421)

    def test_compute_slice_metrics_shape_mismatch(self):
        """Verifies compute_slice_metrics raises ValueError on mismatched slice or mask shapes."""
        slice_a = np.zeros((32, 32), dtype=np.float32)
        slice_b = np.zeros((32, 16), dtype=np.float32)
        norm_a = np.zeros((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)

        with self.assertRaises(ValueError):
            compute_slice_metrics(slice_a, slice_b, norm_a, norm_a, mask, mask)

        with self.assertRaises(ValueError):
            compute_slice_metrics(slice_a, slice_a, norm_a, norm_a, np.zeros((16, 16), dtype=bool), mask)

    def test_compute_slice_metrics_non_finite_values(self):
        """Verifies compute_slice_metrics raises ValueError on NaN or Inf in HU inputs."""
        slice_a = np.zeros((32, 32), dtype=np.float32)
        slice_a[5, 5] = np.nan
        slice_b = np.zeros((32, 32), dtype=np.float32)
        norm = np.zeros((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)

        with self.assertRaises(ValueError):
            compute_slice_metrics(slice_a, slice_b, norm, norm, mask, mask)


if __name__ == "__main__":
    unittest.main()
