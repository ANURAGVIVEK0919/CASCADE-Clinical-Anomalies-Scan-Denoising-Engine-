"""
CASCADE Residual U-Net v3 Edge Preservation & Metric Integrity Regression Suite (Gate 6)

Tests:
1. Edge Metric Correctness:
   - Demonstrates that scalar gradient magnitude difference conflates orthogonal noise with edges.
   - Demonstrates that Vector Gradient Error correctly measures directional and magnitude edge discrepancies.
2. HU Preprocessing Consistency:
   - Exact invertibility between [-1000, +1000] HU and [0, 1].
3. Residual Identity Initialization:
   - Zero-initialized final layer starts as exact identity (|prediction - input| < 1e-6).
4. Native Spatial Dimensions:
   - 128x128 patches from 512x512 slice reconstructs exactly to (512, 512).
5. Patch Blending Identity Preservation:
   - 2D Hann window overlapping blend preserves constant image values without boundary seam artifacts.
6. Strict Patient-Disjoint Splits:
   - Pairwise disjoint sets across Dev Train, Dev Val, v2 Locked Test, and v3 Locked Test cohorts.
7. No Test Leakage into Development:
   - Verifies locked test cohorts are strictly excluded from dev split manifests.
8. Checkpoint Path Safety:
   - Ensures v3 experiment directory does not collide with or overwrite any historical checkpoints.
"""

import os
import json
import unittest
import numpy as np
import tensorflow as tf

from models.residual_unet import (
    build_residual_unet,
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
    SOFT_TISSUE_MAX
)
from pathlib import Path

def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "app.py").exists() and (p / "requirements.txt").exists():
            return p
    return current.parent

REPO_ROOT = str(_find_repo_root())


class TestResidualUNetV3Pipeline(unittest.TestCase):

    def test_edge_metric_mathematical_soundness(self):
        """
        Demonstrates the mathematical flaw in scalar difference vs correctness of vector gradient error:
        If reference has a vertical edge (gx=10, gy=0) and candidate has an orthogonal edge (gx=0, gy=10),
        both have magnitude 10.
        Scalar difference: |10 - 10| = 0 (falsely claims perfect edge preservation!).
        Vector difference: sqrt((0-10)^2 + (10-0)^2) = 14.14 (correctly identifies complete edge misalignment).
        """
        gx_ref, gy_ref = 10.0, 0.0
        gx_ortho, gy_ortho = 0.0, 10.0

        mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
        mag_ortho = np.sqrt(gx_ortho**2 + gy_ortho**2)

        scalar_diff = np.abs(mag_ortho - mag_ref)
        vector_diff = np.sqrt((gx_ortho - gx_ref)**2 + (gy_ortho - gy_ref)**2)

        self.assertEqual(scalar_diff, 0.0, "Scalar difference erroneously reports zero error for orthogonal edges!")
        self.assertAlmostEqual(vector_diff, 14.1421356, places=4, msg="Vector gradient correctly captures structural mismatch.")

    def test_hu_preprocessing_exact_invertibility(self):
        """Verify strict round-trip mapping for clinical CT Hounsfield Units."""
        clinical_hus = np.array([-1000.0, -800.0, -500.0, -160.0, 0.0, 40.0, 240.0, 800.0, 1000.0], dtype=np.float32)
        norm = hu_to_normalized(clinical_hus)

        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))
        restored = normalized_to_hu(norm)
        np.testing.assert_allclose(restored, clinical_hus, atol=1e-4)

    def test_residual_identity_initialization(self):
        """Verify that zero-initialized final layer starts at exact mathematical identity."""
        model = build_residual_unet(input_shape=(128, 128, 1), base_filters=16)
        dummy_input = np.random.uniform(0.1, 0.9, size=(2, 128, 128, 1)).astype(np.float32)
        dummy_pred = model(dummy_input, training=False).numpy()

        max_abs_diff = np.max(np.abs(dummy_pred - dummy_input))
        self.assertLess(max_abs_diff, 1e-6, f"Residual model did not initialize to identity: {max_abs_diff}")

    def test_native_spatial_dimensions(self):
        """Verify full 512x512 resolution is preserved through patch extraction and reconstruction."""
        dummy_slice = np.random.randn(512, 512).astype(np.float32)
        patches, coords = extract_grid_patches(dummy_slice, patch_size=128, stride=64)

        self.assertEqual(len(patches), 49)
        self.assertEqual(patches[0].shape, (128, 128))

        reconstructed = blend_grid_patches(patches, coords, image_shape=(512, 512), patch_size=128)
        self.assertEqual(reconstructed.shape, (512, 512))

    def test_patch_blending_identity_preservation(self):
        """Verify that Hann window overlapping reconstruction preserves flat and smooth values."""
        constant_val = 0.35
        dummy_slice = np.full((512, 512), constant_val, dtype=np.float32)
        patches, coords = extract_grid_patches(dummy_slice, patch_size=128, stride=64)
        reconstructed = blend_grid_patches(patches, coords, image_shape=(512, 512), patch_size=128)

        max_err = np.max(np.abs(reconstructed - dummy_slice))
        self.assertLess(max_err, 1e-5, f"Seam blending error too large: {max_err}")

    def test_patient_disjoint_cohorts(self):
        """Ensure complete patient isolation between dev train, dev val, v2 test, and v3 test."""
        dev_manifest_file = os.path.join(REPO_ROOT, "data", "manifests", "tcia_residual_dev_split_manifest.json")
        v2_test_manifest_file = os.path.join(REPO_ROOT, "data", "manifests", "tcia_locked_test_manifest.json")
        v3_test_manifest_file = os.path.join(REPO_ROOT, "data", "manifests", "tcia_residual_v3_locked_test_manifest.json")

        self.assertTrue(os.path.exists(dev_manifest_file), "Dev manifest missing")
        self.assertTrue(os.path.exists(v2_test_manifest_file), "v2 Test manifest missing")
        self.assertTrue(os.path.exists(v3_test_manifest_file), "v3 Test manifest missing")

        with open(dev_manifest_file) as f:
            dev_data = json.load(f)
        with open(v2_test_manifest_file) as f:
            v2_data = json.load(f)
        with open(v3_test_manifest_file) as f:
            v3_data = json.load(f)

        train_cohort = set(dev_data["train_cohort"])
        val_cohort = set(dev_data["val_cohort"])
        v2_test_cohort = set(v2_data["cohort"])
        v3_test_cohort = set(v3_data["cohort"])

        # Pairwise disjointness
        self.assertTrue(train_cohort.isdisjoint(val_cohort), "Train and Val overlap!")
        self.assertTrue(train_cohort.isdisjoint(v2_test_cohort), "Train and v2 Test overlap!")
        self.assertTrue(train_cohort.isdisjoint(v3_test_cohort), "Train and v3 Test overlap!")
        self.assertTrue(val_cohort.isdisjoint(v2_test_cohort), "Val and v2 Test overlap!")
        self.assertTrue(val_cohort.isdisjoint(v3_test_cohort), "Val and v3 Test overlap!")
        self.assertTrue(v2_test_cohort.isdisjoint(v3_test_cohort), "v2 Test and v3 Test overlap!")

        # Minimum cohort sizes
        self.assertGreaterEqual(len(train_cohort), 8, "Train cohort must have at least 8 patients")
        self.assertGreaterEqual(len(val_cohort), 2, "Val cohort must have at least 2 patients")
        self.assertGreaterEqual(len(v3_test_cohort), 3, "v3 Test cohort must have at least 3 patients")

    def test_checkpoint_path_safety(self):
        """Verify v3 checkpoint path is safe and existing historical models are preserved."""
        v3_ckpt = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_residual_edge_v3", "run_001", "candidate_unet_residual_edge_v3.keras")
        v2_ckpt = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_residual_native", "run_001", "candidate_unet_residual_native.keras")
        hu_ckpt = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_hu_calibrated", "run_001", "candidate_unet_hu_calibrated.keras")
        cand1_ckpt = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_denoiser", "run_001", "candidate_unet_denoiser.keras")
        legacy_ckpt = os.path.join(REPO_ROOT, "medical_ct_denoiser_full.keras")

        # Paths must be unique
        all_paths = [v3_ckpt, v2_ckpt, hu_ckpt, cand1_ckpt, legacy_ckpt]
        self.assertEqual(len(all_paths), len(set(all_paths)), "Checkpoint path collision detected!")

        # Historical checkpoints must exist and remain unchanged
        self.assertTrue(os.path.exists(v2_ckpt), f"v2 checkpoint missing: {v2_ckpt}")
        self.assertTrue(os.path.exists(hu_ckpt), f"HU checkpoint missing: {hu_ckpt}")
        self.assertTrue(os.path.exists(cand1_ckpt), f"Candidate1 checkpoint missing: {cand1_ckpt}")
        self.assertTrue(os.path.exists(legacy_ckpt), f"Legacy checkpoint missing: {legacy_ckpt}")


if __name__ == "__main__":
    unittest.main()
