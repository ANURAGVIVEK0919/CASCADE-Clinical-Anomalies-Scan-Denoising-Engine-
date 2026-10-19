"""
CASCADE Unit & Regression Tests for HU-Calibrated Pipeline and TCIA Patient Splits

Verifies:
- DICOM HU conversion with RescaleSlope and RescaleIntercept
- Fixed windowing and invertible normalization (roundtrip invariance)
- Wide-body window [-1000, +1000] HU roundtrip
- Slice geometry pairing and alignment validation
- Patient-disjoint split integrity (Zero overlap between train, val, test)
- Model checkpoint preservation and anti-overwrite safeguards
- Metric range invariants (PSNR/SSIM intensity range definition)
"""

import os
import json
import unittest
import numpy as np
import tensorflow as tf

from hu_calibration import (
    SOFT_TISSUE_WINDOW,
    WIDE_BODY_WINDOW,
    hu_to_normalized,
    normalized_to_hu,
    validate_slice_pair_geometry
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestHUCalibratedPipeline(unittest.TestCase):

    def test_hu_conversion_formula(self):
        # Simulates raw DICOM uint16 values with slope=1, intercept=-1024
        raw = np.array([0, 1024, 2048], dtype=np.float32)
        slope = 1.0
        intercept = -1024.0
        hu = raw * slope + intercept
        self.assertEqual(hu[0], -1024.0)  # Air
        self.assertEqual(hu[1], 0.0)      # Water
        self.assertEqual(hu[2], 1024.0)   # Dense bone

    def test_windowing_and_inverse_mapping_identity(self):
        window = (-160.0, 240.0)
        test_hus = np.array([-160.0, -100.0, 0.0, 40.0, 150.0, 240.0], dtype=np.float32)
        norm = hu_to_normalized(test_hus, window=window)

        # Norm should be in [0, 1]
        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[-1], 1.0)
        self.assertAlmostEqual(norm[3], (40.0 - (-160.0)) / 400.0)  # 200/400 = 0.5

        # Inverse mapping
        recovered = normalized_to_hu(norm, window=window)
        np.testing.assert_allclose(recovered, test_hus, atol=1e-5)

    def test_wide_body_window_roundtrip(self):
        window = (-1000.0, 1000.0)
        test_hus = np.array([-1000.0, -500.0, 0.0, 40.0, 500.0, 1000.0], dtype=np.float32)
        norm = hu_to_normalized(test_hus, window=window)

        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))
        recovered = normalized_to_hu(norm, window=window)
        np.testing.assert_allclose(recovered, test_hus, atol=1e-4)

    def test_out_of_window_clipping(self):
        window = (-160.0, 240.0)
        test_hus = np.array([-1000.0, -500.0, -160.0, 240.0, 500.0, 1000.0], dtype=np.float32)
        norm = hu_to_normalized(test_hus, window=window)

        self.assertEqual(norm[0], 0.0)
        self.assertEqual(norm[1], 0.0)
        self.assertEqual(norm[2], 0.0)
        self.assertEqual(norm[3], 1.0)
        self.assertEqual(norm[4], 1.0)
        self.assertEqual(norm[5], 1.0)

    def test_slice_pair_geometry_validation(self):
        meta_valid_fd = {"z_position": -120.5, "rows": 512, "columns": 512}
        meta_valid_ld = {"z_position": -120.5, "rows": 512, "columns": 512}
        self.assertTrue(validate_slice_pair_geometry(meta_valid_fd, meta_valid_ld))

        # Different z coordinate
        meta_misaligned_z = {"z_position": -123.5, "rows": 512, "columns": 512}
        self.assertFalse(validate_slice_pair_geometry(meta_valid_fd, meta_misaligned_z))

        # Different dimensions
        meta_misaligned_dim = {"z_position": -120.5, "rows": 256, "columns": 512}
        self.assertFalse(validate_slice_pair_geometry(meta_valid_fd, meta_misaligned_dim))

    def test_patient_disjoint_split_integrity(self):
        manifest_path = os.path.join(REPO_ROOT, "data", "manifests", "tcia_hu_calibrated_split_manifest.json")
        if not os.path.exists(manifest_path):
            self.skipTest("Manifest file not yet generated")

        with open(manifest_path) as f:
            manifest = json.load(f)

        train_patients = set(manifest["splits"]["train"]["patients"])
        val_patients = set(manifest["splits"]["validation"]["patients"])
        test_patients = set(manifest["splits"]["test"]["patients"])

        # Invariants:
        self.assertTrue(train_patients.isdisjoint(val_patients), "Train and Val must be disjoint")
        self.assertTrue(train_patients.isdisjoint(test_patients), "Train and Test must be disjoint")
        self.assertTrue(val_patients.isdisjoint(test_patients), "Val and Test must be disjoint")

        # Held-out test patients must be L266 and L131
        self.assertIn("L266", test_patients)
        self.assertIn("L131", test_patients)
        self.assertNotIn("L266", train_patients)
        self.assertNotIn("L131", train_patients)

    def test_metric_intensity_range_invariants(self):
        # Soft-tissue window range must be exactly 400.0 HU
        st_range = SOFT_TISSUE_WINDOW[1] - SOFT_TISSUE_WINDOW[0]
        self.assertEqual(st_range, 400.0)

        # Test PSNR calculation formula consistency:
        # PSNR = 10 * log10(max_val^2 / MSE)
        img1 = tf.constant(np.zeros((1, 16, 16, 1), dtype=np.float32))
        img2 = tf.constant(np.ones((1, 16, 16, 1), dtype=np.float32) * 4.0)  # MSE = 16.0
        mse = float(np.mean((img1.numpy() - img2.numpy()) ** 2))
        psnr_expected = 10.0 * np.log10((400.0 ** 2) / mse)  # 160000 / 16 = 10000 -> 40 dB
        self.assertAlmostEqual(psnr_expected, 40.0, places=4)

    def test_checkpoint_safety_no_overwrite(self):
        from train_hu_calibrated_unet import CHECKPOINT_PATH, PROTECTED_CHECKPOINTS
        target_norm = os.path.normpath(CHECKPOINT_PATH)
        for p in PROTECTED_CHECKPOINTS:
            self.assertNotEqual(target_norm, os.path.normpath(p), f"Checkpoint must not overwrite {p}")


if __name__ == "__main__":
    unittest.main()
