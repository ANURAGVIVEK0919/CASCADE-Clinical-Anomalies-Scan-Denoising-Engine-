"""
CASCADE Dataset Evaluation Script

Evaluates:
1. Diagnostic Model Organ Classification (Chest, Brain, Abdomen)
2. Diagnostic Model Candidate ROI Localization Rate across severity-tagged folders

Configurable via CLI arguments and environment variables without hardcoded machine paths.
"""

import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import get_model, get_diagnostic_model, process_image, run_diagnostic_analyzer


def evaluate_directory(data_dir=None, max_images=50):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not data_dir:
        data_dir = os.environ.get("CASCADE_DATA_DIR", os.path.join(base_dir, "data", "separated_scans"))

    print("=" * 60)
    print("  CASCADE DIAGNOSTIC EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Target Directory: {data_dir}")

    # Load diagnostic model
    m_diag = get_diagnostic_model()
    if m_diag is None:
        print("[!] ERROR: Diagnostic ResNet model failed to load!")
        return

    search_pattern = os.path.join(data_dir, "**", "noisy", "*.png")
    all_files = glob.glob(search_pattern, recursive=True)
    if not all_files:
        search_pattern = os.path.join(data_dir, "**", "*.png")
        all_files = glob.glob(search_pattern, recursive=True)

    roi_stats = {
        'high_risk': {'total': 0, 'candidate_roi_flagged': 0},
        'medium_risk': {'total': 0, 'candidate_roi_flagged': 0},
        'normal': {'total': 0, 'candidate_roi_flagged': 0}
    }

    print(f"Found {len(all_files)} candidate images in {data_dir}.")
    if not all_files:
        print("[!] No images found to evaluate.")
        return {
            "total_evaluated": 0,
            "organ_correct": 0,
            "accuracy": 0.0,
            "roi_stats": roi_stats
        }

    if max_images > 0 and len(all_files) > max_images:
        import random
        random.seed(42)
        all_files = random.sample(all_files, max_images)
        print(f"Sampled {len(all_files)} images for reproducible evaluation.")

    organs = ["chest", "brain", "abdomen"]
    organ_correct = 0
    total_evaluated = 0

    for i, file_path in enumerate(all_files):
        # Ground truth organ and severity folder from path
        norm_path = file_path.lower().replace("\\", "/")
        actual_organ = None
        for o in organs:
            if f"/{o}/" in norm_path or f"_{o}_" in norm_path:
                actual_organ = o
                break
        if not actual_organ:
            continue

        folder_severity = "normal"
        if "high_risk" in norm_path:
            folder_severity = "high_risk"
        elif "medium_risk" in norm_path:
            folder_severity = "medium_risk"

        # 1. Process image
        noisy, denoised, diff, metrics, meta = process_image(file_path)

        # 2. Run genuine diagnostic inference
        input_tensor = np.expand_dims(denoised, axis=0)
        class_pred, bbox_pred = m_diag.predict(input_tensor, verbose=0)

        # Real model predicted organ
        class_idx = int(np.argmax(class_pred[0]))
        class_map = {0: 'chest', 1: 'brain', 2: 'abdomen'}
        pred_organ = class_map.get(class_idx, 'chest')

        if pred_organ == actual_organ:
            organ_correct += 1
        total_evaluated += 1

        # Real model candidate ROI
        ymin, xmin, ymax, xmax = bbox_pred[0].tolist()
        bbox = [ymin, xmin, ymax, xmax]

        # Real model analyzer
        diagnosis = run_diagnostic_analyzer(denoised, actual_organ, bbox, predicted_organ=pred_organ, filename=os.path.basename(file_path))
        severity = diagnosis['severity']

        if folder_severity in roi_stats:
            roi_stats[folder_severity]['total'] += 1
            if severity == "Candidate ROI Detected":
                roi_stats[folder_severity]['candidate_roi_flagged'] += 1

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    accuracy = 0.0
    if total_evaluated > 0:
        accuracy = (organ_correct / total_evaluated) * 100.0
        print(f"Total Scans Evaluated: {total_evaluated}")
        print(f"Organ Classification Accuracy: {accuracy:.2f}% ({organ_correct}/{total_evaluated})")
        print("-" * 60)
        print("Candidate ROI Detection Rate by Folder Category:")
        for cat, stats in roi_stats.items():
            if stats['total'] > 0:
                flag_pct = (stats['candidate_roi_flagged'] / stats['total']) * 100.0
                print(f"  Category {cat.upper():<12}: {stats['candidate_roi_flagged']}/{stats['total']} flagged ({flag_pct:.1f}%)")
    else:
        print("[!] No matching organ scans could be evaluated.")

    return {
        "total_evaluated": total_evaluated,
        "organ_correct": organ_correct,
        "accuracy": accuracy,
        "roi_stats": roi_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CASCADE Models on Scan Directory")
    parser.add_argument("--data-dir", default=None, help="Directory containing scans")
    parser.add_argument("--max-images", type=int, default=30, help="Maximum number of images to evaluate")
    args = parser.parse_args()

    evaluate_directory(data_dir=args.data_dir, max_images=args.max_images)
