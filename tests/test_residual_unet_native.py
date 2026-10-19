"""
Unit and Regression Tests for CASCADE Residual U-Net v2 and Native Resolution Pipeline
"""

import os
import json
import unittest
import numpy as np

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


class TestResidualUNetNativePipeline(unittest.TestCase):

    def test_hu_conversion_and_inverse_mapping(self):
        """Test exact invertibility of HU preprocessing over clinical ranges."""
        test_hus = np.array([-1000.0, -500.0, -160.0, 0.0, 40.0, 240.0, 1000.0], dtype=np.float32)
        norm = hu_to_normalized(test_hus)

        self.assertTrue(np.all(norm >= 0.0))
        self.assertTrue(np.all(norm <= 1.0))
        self.assertAlmostEqual(float(norm[0]), 0.0, places=5)    # -1000 HU (air)
        self.assertAlmostEqual(float(norm[3]), 0.5, places=5)    # 0 HU (water)
        self.assertAlmostEqual(float(norm[6]), 1.0, places=5)    # 1000 HU (bone)

        restored_hus = normalized_to_hu(norm)
        np.testing.assert_allclose(restored_hus, test_hus, atol=1e-4)

    def test_identity_initialization(self):
        """Verify that zero-initialized final layer yields exact identity at epoch 0."""
        model = build_residual_unet(input_shape=(128, 128, 1), base_filters=16)
        dummy_input = np.random.uniform(0.1, 0.9, size=(2, 128, 128, 1)).astype(np.float32)
        dummy_pred = model(dummy_input, training=False).numpy()

        max_abs_diff = np.max(np.abs(dummy_pred - dummy_input))
        self.assertLess(max_abs_diff, 1e-6, "Residual model must initialize as exact identity.")

    def test_preservation_of_spatial_dimensions(self):
        """Verify that patch extraction and reconstruction preserves (512, 512) dimensions."""
        img = np.random.randn(512, 512).astype(np.float32)
        patches, coords = extract_grid_patches(img, patch_size=128, stride=64)

        # Grid steps: (512 - 128)/64 + 1 = 7 steps each direction -> 49 patches
        self.assertEqual(len(patches), 49)
        self.assertEqual(patches[0].shape, (128, 128))

        reconstructed = blend_grid_patches(patches, coords, image_shape=(512, 512), patch_size=128)
        self.assertEqual(reconstructed.shape, (512, 512))

    def test_overlapping_patch_reconstruction_identity(self):
        """Verify that 2D Hann window blending perfectly reconstructs a constant or smooth image."""
        const_val = 0.42
        img = np.full((512, 512), const_val, dtype=np.float32)
        patches, coords = extract_grid_patches(img, patch_size=128, stride=64)
        reconstructed = blend_grid_patches(patches, coords, image_shape=(512, 512), patch_size=128)

        # Difference should be near machine epsilon everywhere
        max_err = np.max(np.abs(reconstructed - img))
        self.assertLess(max_err, 1e-5, f"Reconstruction failed identity blend with error: {max_err}")

    def test_patient_disjoint_splits(self):
        """Ensure strict separation: no patient overlap between train, val, and locked test."""
        dev_path = os.path.join(REPO_ROOT, "data", "manifests", "tcia_residual_dev_split_manifest.json")
        test_path = os.path.join(REPO_ROOT, "data", "manifests", "tcia_locked_test_manifest.json")

        self.assertTrue(os.path.exists(dev_path), "Dev split manifest missing")
        self.assertTrue(os.path.exists(test_path), "Locked test manifest missing")

        with open(dev_path) as f:
            dev_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)

        train_cohort = set(dev_data["train_cohort"])
        val_cohort = set(dev_data["val_cohort"])
        test_cohort = set(test_data["cohort"])

        self.assertTrue(train_cohort.isdisjoint(val_cohort), "Train and Val cohorts overlap!")
        self.assertTrue(train_cohort.isdisjoint(test_cohort), "Train and Locked Test cohorts overlap!")
        self.assertTrue(val_cohort.isdisjoint(test_cohort), "Val and Locked Test cohorts overlap!")
        self.assertEqual(len(test_cohort), 3, "Locked test cohort must contain 3 patients")

    def test_safe_checkpoint_paths(self):
        """Verify candidate checkpoint path never collides with historical models."""
        from train_residual_unet_native import CHECKPOINT_PATH

        legacy_path = os.path.join(REPO_ROOT, "medical_ct_denoiser_full.keras")
        candidate1_path = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_denoiser", "run_001", "candidate_unet_denoiser.keras")
        candidate_hu_path = os.path.join(REPO_ROOT, "artifacts", "experiments", "unet_denoiser_hu", "run_001", "candidate_unet_denoiser_hu.keras")

        norm_ckpt = os.path.abspath(CHECKPOINT_PATH)
        self.assertNotEqual(norm_ckpt, os.path.abspath(legacy_path))
        self.assertNotEqual(norm_ckpt, os.path.abspath(candidate1_path))
        self.assertNotEqual(norm_ckpt, os.path.abspath(candidate_hu_path))
        self.assertIn("unet_residual_native", norm_ckpt)

    def test_metric_intensity_range_consistency(self):
        """Verify soft-tissue and full HU ranges are non-empty and well-ordered."""
        self.assertLess(HU_MIN, HU_MAX)
        self.assertLess(SOFT_TISSUE_MIN, SOFT_TISSUE_MAX)
        self.assertGreaterEqual(SOFT_TISSUE_MIN, HU_MIN)
        self.assertLessEqual(SOFT_TISSUE_MAX, HU_MAX)

        # Check mask functionality
        sample_hu = np.array([-1000.0, -160.0, 0.0, 100.0, 240.0, 500.0])
        st_mask = get_soft_tissue_mask(sample_hu)
        self.assertEqual(list(st_mask), [False, True, True, True, True, False])


if __name__ == "__main__":
    unittest.main()
