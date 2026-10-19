"""
CASCADE Residual U-Net v2 Architecture and Native Patch Reconstruction Utilities

Implements:
1. Residual U-Net formulation:
   prediction = input + residual_correction
   With zero-initialized final conv layer (identity initialization).
2. Native patch extraction and overlapping patch reconstruction with 2D Hann window blending.
3. Fixed, documented HU preprocessing:
   Linear mapping [-1000, +1000] HU -> [0, 1].
   Inverse mapping [0, 1] -> [-1000, +1000] HU.
   Abdominal soft-tissue window [-160, +240] HU.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Global HU Calibration Parameters
HU_MIN = -1000.0
HU_MAX = 1000.0
HU_RANGE = HU_MAX - HU_MIN  # 2000.0 HU

SOFT_TISSUE_MIN = -160.0
SOFT_TISSUE_MAX = 240.0
BODY_MASK_THRESHOLD = -500.0  # Anatomical tissue vs surrounding air/bed


def hu_to_normalized(hu_image: np.ndarray) -> np.ndarray:
    """
    Fixed, deterministic HU mapping from [-1000, +1000] HU to [0, 1].
    Preserves all clinical tissue types without arbitrary per-image min-max scaling.
    """
    clipped = np.clip(hu_image, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / HU_RANGE


def normalized_to_hu(norm_image: np.ndarray, clip_to_bounds: bool = False) -> np.ndarray:
    """
    Exact inverse mapping from [0, 1] to [-1000, +1000] HU.
    If clip_to_bounds is True, clamps normalized input to [0, 1] before scaling.
    """
    if clip_to_bounds:
        norm_image = np.clip(norm_image, 0.0, 1.0)
    return norm_image * HU_RANGE + HU_MIN


def get_body_contour_mask(hu_image: np.ndarray, threshold: float = BODY_MASK_THRESHOLD) -> np.ndarray:
    """
    Derives anatomical body mask from calibrated HU data (> threshold).
    """
    return hu_image > threshold


def get_soft_tissue_mask(hu_image: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask for abdominal soft tissue window [-160, +240] HU.
    """
    return (hu_image >= SOFT_TISSUE_MIN) & (hu_image <= SOFT_TISSUE_MAX)


def build_residual_unet(
    input_shape: tuple = (128, 128, 1),
    base_filters: int = 32,
    kernel_size: tuple = (3, 3)
) -> models.Model:
    """
    Builds Residual U-Net v2 with zero-initialized final residual layer.

    Residual formulation:
      prediction = input + residual_correction
      residual_correction = Conv2D(1, (1, 1), kernel_initializer='zeros', bias_initializer='zeros')

    Property:
      At initialization (epoch 0), residual_correction == 0, so prediction == input.
      The model starts as an exact identity mapping and only learns constructive denoising deltas.
    """
    inputs = layers.Input(shape=input_shape, name="low_dose_input")

    # Encoder Block 1
    c1 = layers.Conv2D(base_filters, kernel_size, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(base_filters, kernel_size, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Encoder Block 2
    c2 = layers.Conv2D(base_filters * 2, kernel_size, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(base_filters * 2, kernel_size, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(base_filters * 4, kernel_size, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(base_filters * 4, kernel_size, activation="relu", padding="same")(c3)

    # Decoder Block 2
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(base_filters * 2, kernel_size, activation="relu", padding="same")(u2)
    c4 = layers.Conv2D(base_filters * 2, kernel_size, activation="relu", padding="same")(c4)

    # Decoder Block 1
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(base_filters, kernel_size, activation="relu", padding="same")(u1)
    c5 = layers.Conv2D(base_filters, kernel_size, activation="relu", padding="same")(c5)

    # Residual Correction Layer (ZERO INITIALIZED)
    residual_correction = layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="residual_correction"
    )(c5)

    # Prediction = Input + Residual Correction
    predictions = layers.Add(name="residual_add")([inputs, residual_correction])

    return models.Model(inputs=inputs, outputs=predictions, name="residual_unet_v2")


def extract_grid_patches(
    image: np.ndarray,
    patch_size: int = 128,
    stride: int = 64
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Extracts 2D grid patches with overlap from image shape (H, W).
    Returns (patches array [N, patch_size, patch_size], coordinates [(r, c)]).
    """
    H, W = image.shape[:2]
    coords = []
    patches = []

    r_steps = list(range(0, H - patch_size + 1, stride))
    if r_steps[-1] != H - patch_size:
        r_steps.append(H - patch_size)

    c_steps = list(range(0, W - patch_size + 1, stride))
    if c_steps[-1] != W - patch_size:
        c_steps.append(W - patch_size)

    for r in r_steps:
        for c in c_steps:
            patch = image[r:r + patch_size, c:c + patch_size]
            patches.append(patch)
            coords.append((r, c))

    return np.array(patches), coords


def make_2d_hann_window(patch_size: int = 128) -> np.ndarray:
    """
    Generates a 2D Hann window for smooth seam-free blending.
    """
    w1d = np.hanning(patch_size)
    w1d = np.clip(w1d, 1e-3, 1.0)
    w2d = np.outer(w1d, w1d)
    return w2d.astype(np.float32)


def blend_grid_patches(
    patches: np.ndarray,
    coords: list[tuple[int, int]],
    image_shape: tuple[int, int] = (512, 512),
    patch_size: int = 128
) -> np.ndarray:
    """
    Blends overlapping patches using a 2D Hann window to reconstruct full-resolution image.
    Eliminates seam artifacts at patch boundaries.
    """
    H, W = image_shape
    accum = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    window = make_2d_hann_window(patch_size)

    for patch, (r, c) in zip(patches, coords):
        p = patch.squeeze()
        accum[r:r + patch_size, c:c + patch_size] += p * window
        weights[r:r + patch_size, c:c + patch_size] += window

    # Normalize by accumulated weights with safe numerical precision
    weights = np.maximum(weights, 1e-12)
    reconstructed = accum / weights
    return reconstructed


def denoise_native_slice(
    model: models.Model,
    low_dose_norm_512: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    batch_size: int = 16
) -> np.ndarray:
    """
    Full inference on a native 512x512 slice using overlapping patches and Hann blending.
    No image downsampling or resizing is performed.
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
