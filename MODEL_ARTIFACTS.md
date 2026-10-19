# CASCADE Neural Model Artifacts & Packaging Specification

This document defines the authoritative file specifications, cryptographic checksums, input/output tensor contracts, and packaging requirements for CASCADE's deep learning models.

---

## 1. Model Artifact Inventory

| Model Name | Expected Location | Exact Size (Bytes) | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net CT Denoiser** | `medical_ct_denoiser_full.keras` | `23,388,275` | 22.30 MB | `ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842` |
| **Diagnostic ResNet50V2** | `medical_ct_diagnostic_model.keras` | `256,877,707` | 244.98 MB | `a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41` |

> **Note on Duplicate Paths**: The directory `output/` contains secondary copies of both model files (`output/medical_ct_denoiser_full.keras` and `output/medical_ct_diagnostic_model.keras`). These copies have been verified to be byte-for-byte identical to the root models with matching SHA-256 hashes.

---

## 2. Input/Output Contracts & Tensor Semantics

### A. U-Net CT Denoiser (`medical_ct_denoiser_full.keras`)
* **Architecture**: Symmetric Convolutional U-Net Autoencoder with skip connections.
* **Input Tensor**: `(batch_size, 256, 256, 1)` (float32).
* **Preprocessing Range**: `[0.0, 1.0]`. Grayscale CT slices normalized by dividing pixel intensities by 255.0 (or min-max HU windowing).
* **Degenerate Slices**: Uniform or zero-range slices are mapped safely to `np.zeros_like`.
* **Output Projection Layer**: `Conv2D(1, (1, 1), activation='sigmoid')`.
* **Output Tensor**: `(batch_size, 256, 256, 1)` (float32, range `[0.0, 1.0]`).
* **Postprocessing**: Scaled back to 8-bit `[0, 255]` and bilinearly interpolated to the original DICOM/PNG resolution.

### B. Diagnostic Multi-Task Model (`medical_ct_diagnostic_model.keras`)
* **Architecture**: ResNet50V2 backbone with dual output heads.
* **Input Tensor**: `(batch_size, 256, 256, 1)` (float32, range `[0.0, 1.0]`).
* **Dual Outputs**:
  1. **Organ Classification**: `(batch_size, 3)` (float32, softmax probabilities over `Chest`, `Brain`, `Abdomen`). Labeled as *Uncalibrated Model Scores*.
  2. **Candidate ROI Coordinates**: `(batch_size, 4)` (float32, normalized bounding box `[ymin, xmin, ymax, xmax]` in range `[0.0, 1.0]`).
* **Explainability**: Saliency Grad-CAM maps are generated from the convolutional layers of the ResNet backbone to visualize organ features (not pathology localization).

---

## 3. Version Control & Packaging Rules

1. **Git Exclusions**:
   * All `.keras`, `.h5`, and checkpoint weight files are excluded from Git tracking via `.gitignore` (lines 131–132).
   * A clean `git clone` or fresh checkout does **not** include these binary weights.
2. **Docker Packaging**:
   * `.dockerignore` does not exclude `.keras` files so that local builds (`COPY . .`) can package models into container images when models are placed in the repository root.
   * A Docker build initiated from a bare checkout without model files will build the image, but the container will operate in degraded mode without denoising or organ classification.

---

## 4. Developer Setup Instructions

Before running the Flask server, DICOM pipeline, or integration test suite:

1. Obtain the verified `.keras` weight files from your organization's approved model registry or internal storage vault.
2. Place the two files in the project root directory:
   * `./medical_ct_denoiser_full.keras`
   * `./medical_ct_diagnostic_model.keras`
3. Verify file integrity using SHA-256:
   ```powershell
   Get-FileHash medical_ct_denoiser_full.keras, medical_ct_diagnostic_model.keras -Algorithm SHA256
   ```
4. Verify models load correctly:
   ```powershell
   .\venv\Scripts\python.exe -m unittest tests.test_integration -k "test_real_model" -v
   ```

---

## 5. Missing Model Error Behavior

* **DICOM Batch Pipeline (`data_pipeline.py`)**: Raises an explicit, actionable `FileNotFoundError`:
  `U-Net denoiser model weights not found at 'medical_ct_denoiser_full.keras'. Training model weights are required for batch DICOM pipeline execution.`
* **Flask Inference Route (`app.py`)**: Returns `None` for missing models and safely skips neural inference without crashing, returning clear status messages.
