#!/usr/bin/env python3
"""
CASCADE — Model Artifact Verification Script (ARM64 Backend)

Verifies byte size and SHA-256 digests of all required neural model weights
before application startup on OCI Ampere A1 (or any standalone host).

Expected Artifacts:
1. medical_ct_denoiser_full.keras:
   - Size: 23,388,275 bytes
   - SHA-256: ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842
2. medical_ct_diagnostic_model.keras:
   - Size: 256,877,707 bytes
   - SHA-256: a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41
3. artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras:
   - Size: 1,936,708 bytes
   - SHA-256: bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea
"""

import os
import sys
import hashlib

EXPECTED_ARTIFACTS = {
    'medical_ct_denoiser_full.keras': {
        'path': 'medical_ct_denoiser_full.keras',
        'expected_size': 23388275,
        'expected_sha256': 'ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842',
        'mandatory': True
    },
    'medical_ct_diagnostic_model.keras': {
        'path': 'medical_ct_diagnostic_model.keras',
        'expected_size': 256877707,
        'expected_sha256': 'a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41',
        'mandatory': True
    },
    'candidate_unet_residual_edge_v3.keras': {
        'path': os.path.join(
            'artifacts', 'experiments', 'unet_residual_edge_v3', 'run_001',
            'candidate_unet_residual_edge_v3.keras'
        ),
        'expected_size': 1936708,
        'expected_sha256': 'bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea',
        'mandatory': False  # Research candidate; application falls back safely if absent
    }
}


def compute_sha256(filepath):
    """Computes SHA-256 hash using streaming chunks to minimize memory usage."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifacts(base_dir='.'):
    """Verifies all model artifacts present under base_dir."""
    print("=" * 70)
    print("CASCADE ARM64 Backend — Model Artifact Integrity Verification")
    print(f"Base Directory: {os.path.abspath(base_dir)}")
    print("=" * 70)

    all_passed = True

    for name, spec in EXPECTED_ARTIFACTS.items():
        rel_path = spec['path']
        full_path = os.path.join(base_dir, rel_path)
        is_mandatory = spec['mandatory']
        exp_size = spec['expected_size']
        exp_sha = spec['expected_sha256']

        print(f"\n[*] Checking: {name}")
        print(f"    Path: {full_path}")

        if not os.path.exists(full_path):
            if is_mandatory:
                print(f"    [FAIL] Mandatory model artifact NOT FOUND.")
                all_passed = False
            else:
                print(f"    [WARN] Optional research candidate NOT FOUND. Fallback mode will be active.")
            continue

        actual_size = os.path.getsize(full_path)
        if actual_size != exp_size:
            print(f"    [FAIL] Size mismatch: got {actual_size} bytes, expected {exp_size} bytes.")
            all_passed = False
            continue
        print(f"    [PASS] Size: {actual_size:,} bytes")

        actual_sha = compute_sha256(full_path)
        if actual_sha.lower() != exp_sha.lower():
            print(f"    [FAIL] SHA-256 mismatch!")
            print(f"           Got:      {actual_sha}")
            print(f"           Expected: {exp_sha}")
            all_passed = False
            continue
        print(f"    [PASS] SHA-256: {actual_sha}")

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All verified model artifacts match expected digests.")
        print("=" * 70)
        return 0
    else:
        print("[ERROR] One or more mandatory model artifacts failed verification.")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '.'
    sys.exit(verify_artifacts(base))
