"""
Comprehensive validation script to run inference on dataset images.
Tests:
1. Denoising performance (PSNR/SSIM improvements on paired clean/noisy scans)
2. Diagnostic organ classification accuracy (Chest vs Brain vs Abdomen)
3. Candidate ROI detection distributions across folder categories

Configurable via CLI arguments and environment variables.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_image, get_model, get_diagnostic_model, run_diagnostic_analyzer


def run_validation(data_dir=None, sample_size_per_organ=50, output_path=None, overwrite=False):
    print("=" * 65)
    print("  CASCADE: DATASET-WIDE MODEL VALIDATION SUITE")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not data_dir:
        data_dir = os.environ.get("CASCADE_DATA_DIR", os.path.join(base_dir, "data", "separated_scans"))

    default_output = os.path.join(base_dir, "docs", "validation_report.txt")
    target_output_path = output_path if output_path is not None else default_output

    # Pre-load models
    model = get_model()
    diag_model = get_diagnostic_model()

    if model is None:
        print("[!] Warning: Denoising U-Net model could not be loaded. Classical smoothing baseline will be used.")
    else:
        print("[+] Denoising U-Net loaded successfully into memory.")

    if diag_model is None:
        print("[!] Error: Diagnostic ResNet model could not be loaded!")
        return None
    print("[+] Diagnostic ResNet loaded successfully into memory.")

    organs = ["chest", "brain", "abdomen"]
    folders = ["normal", "medium_risk", "high_risk"]

    total_images = 0
    correct_organ_preds = 0

    # Accumulators for paired denoising metrics
    paired_images_count = 0
    psnr_noisy_sum = 0.0
    psnr_restored_sum = 0.0
    ssim_noisy_sum = 0.0
    ssim_restored_sum = 0.0
    improvement_sum = 0.0

    # Candidate ROI localization distribution: actual_folder -> candidate_roi_status -> count
    severity_dist = {
        folder: {"Candidate ROI Detected": 0, "No Candidate ROI Flagged": 0}
        for folder in folders
    }

    # Organ prediction counts: actual_organ -> predicted_organ -> count
    organ_matrix = {o: {pred_o: 0 for pred_o in organs} for o in organs}

    import random
    random.seed(42)

    file_list = []
    for organ in organs:
        organ_files = []
        for folder in folders:
            noisy_dir = os.path.join(data_dir, organ, folder, "noisy")
            if not os.path.exists(noisy_dir):
                continue
            files = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                organ_files.append({
                    'file_path': os.path.join(noisy_dir, f),
                    'filename': f,
                    'actual_organ': organ,
                    'actual_severity': folder
                })

        sample_size = min(sample_size_per_organ, len(organ_files)) if sample_size_per_organ > 0 else len(organ_files)
        if organ_files:
            sampled = random.sample(organ_files, sample_size)
            file_list.extend(sampled)
            print(f"[*] Sampled {sample_size} scans for organ: {organ.upper()} (out of {len(organ_files)})")
        else:
            print(f"[!] No scans found for organ: {organ.upper()} in {data_dir}")

    total_files = len(file_list)
    print(f"[*] Total sampled scans to validate: {total_files}")
    if total_files == 0:
        print("[!] No images found to validate.")
        return {
            "total_images": 0,
            "paired_images_count": 0,
            "organ_accuracy": 0.0,
            "report_written": False
        }

    print("[*] Starting batch inference on sampled scans...")
    start_time = time.time()

    for idx, item in enumerate(file_list):
        filepath = item['file_path']
        actual_organ = item['actual_organ']
        actual_severity = item['actual_severity']

        # 1. Process image (Denoising & scientific metrics)
        noisy_arr, denoised_arr, diff_map, metrics, _ = process_image(filepath)

        # 2. Run diagnostic ResNet
        input_tensor = np.expand_dims(denoised_arr, axis=0)
        class_pred, bbox_pred = diag_model.predict(input_tensor, verbose=0)

        # Class mapping
        class_idx = int(np.argmax(class_pred[0]))
        class_map = {0: 'chest', 1: 'brain', 2: 'abdomen'}
        pred_organ = class_map.get(class_idx, 'chest')

        # Update organ classification metrics
        organ_matrix[actual_organ][pred_organ] += 1
        if pred_organ == actual_organ:
            correct_organ_preds += 1

        # 3. Diagnostic Analyzer (Candidate ROI Localization)
        ymin, xmin, ymax, xmax = bbox_pred[0].tolist()
        bbox = [ymin, xmin, ymax, xmax]
        diagnosis = run_diagnostic_analyzer(denoised_arr, actual_organ, bbox, pred_organ)
        pred_severity = diagnosis.get('severity', 'No Candidate ROI Flagged')

        if actual_severity in severity_dist and pred_severity in severity_dist[actual_severity]:
            severity_dist[actual_severity][pred_severity] += 1

        # Safe accumulation of paired denoising metrics
        if metrics.get('psnr_noisy') is not None and metrics.get('psnr_restored') is not None:
            psnr_noisy_sum += metrics['psnr_noisy']
            psnr_restored_sum += metrics['psnr_restored']
            ssim_noisy_sum += metrics['ssim_noisy']
            ssim_restored_sum += metrics['ssim_restored']
            if metrics.get('improvement') is not None:
                improvement_sum += metrics['improvement']
            paired_images_count += 1

        total_images += 1

        if total_images % 25 == 0 or total_images == total_files:
            elapsed = time.time() - start_time
            img_per_sec = total_images / max(1e-5, elapsed)
            print(f"    Processed {total_images}/{total_files} | Organ Accuracy: {100.0 * correct_organ_preds / total_images:.1f}% | Speed: {img_per_sec:.1f} img/sec")

    total_time = time.time() - start_time

    # Calculate Denoising Averages safely
    if paired_images_count > 0:
        avg_psnr_noisy = psnr_noisy_sum / paired_images_count
        avg_psnr_restored = psnr_restored_sum / paired_images_count
        avg_ssim_noisy = ssim_noisy_sum / paired_images_count
        avg_ssim_restored = ssim_restored_sum / paired_images_count
        avg_improvement = improvement_sum / paired_images_count
        psnr_gain = avg_psnr_restored - avg_psnr_noisy
        ssim_gain = avg_ssim_restored - avg_ssim_noisy
    else:
        avg_psnr_noisy = avg_psnr_restored = avg_ssim_noisy = avg_ssim_restored = avg_improvement = psnr_gain = ssim_gain = None

    organ_accuracy = (correct_organ_preds / max(1, total_images)) * 100.0

    # Write report safely
    report_written = False
    if target_output_path:
        if os.path.exists(target_output_path) and not overwrite:
            print(f"[!] Warning: Output file '{target_output_path}' already exists and overwrite=False. Skipping report file write (pass --overwrite to allow).")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(target_output_path)), exist_ok=True)
            with open(target_output_path, "w", encoding="utf-8") as f:
                f.write("==========================================================\n")
                f.write("      CASCADE RESTORATION & DIAGNOSTICS REPORT\n")
                f.write("==========================================================\n")
                f.write(f"Total Scans Processed: {total_images}\n")
                f.write(f"Paired Reference Scans: {paired_images_count}\n")
                f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
                f.write(f"Average Processing Speed: {total_images / max(1e-5, total_time):.2f} images/sec\n\n")

                f.write("----------------------------------------------------------\n")
                f.write("1. DENOISING PERFORMANCE METRICS\n")
                f.write("----------------------------------------------------------\n")
                if paired_images_count > 0:
                    f.write(f"Average PSNR (Noisy Baseline): {avg_psnr_noisy:.2f} dB\n")
                    f.write(f"Average PSNR (Denoised U-Net):  {avg_psnr_restored:.2f} dB\n")
                    f.write(f"Average PSNR Gain:              +{psnr_gain:.2f} dB\n\n")
                    f.write(f"Average SSIM (Noisy Baseline): {avg_ssim_noisy:.4f}\n")
                    f.write(f"Average SSIM (Denoised U-Net):  {avg_ssim_restored:.4f}\n")
                    f.write(f"Average SSIM Gain:              +{ssim_gain:.4f}\n\n")
                    f.write(f"Average Signal Quality Improvement: {avg_improvement:.2f}%\n\n")
                else:
                    f.write("No paired ground-truth clean references available for these scans.\n\n")

                f.write("----------------------------------------------------------\n")
                f.write("2. ORGAN CLASSIFICATION ACCURACY\n")
                f.write("----------------------------------------------------------\n")
                f.write(f"Overall Classification Accuracy: {organ_accuracy:.2f}%\n\n")
                f.write("Confusion Matrix:\n")
                f.write("Actual \\ Predicted |   Chest  |   Brain  |  Abdomen \n")
                for o in organs:
                    c = organ_matrix[o]["chest"]
                    b = organ_matrix[o]["brain"]
                    a = organ_matrix[o]["abdomen"]
                    f.write(f"  {o.capitalize():<16} |   {c:5d}  |   {b:5d}  |   {a:7d}\n")

                f.write("\n----------------------------------------------------------\n")
                f.write("3. CANDIDATE ROI LOCALIZATION FLAG RATE\n")
                f.write("----------------------------------------------------------\n")
                for folder, counts in severity_dist.items():
                    tot = sum(counts.values())
                    roi_det = counts.get("Candidate ROI Detected", 0)
                    rate = (roi_det / tot * 100.0) if tot > 0 else 0.0
                    f.write(f"Folder '{folder}': {roi_det}/{tot} flagged with candidate ROI ({rate:.1f}%)\n")

            report_written = True
            print(f"\n[+] Validation report successfully written to: {target_output_path}")

    return {
        "total_images": total_images,
        "paired_images_count": paired_images_count,
        "organ_accuracy": organ_accuracy,
        "organ_matrix": organ_matrix,
        "severity_dist": severity_dist,
        "psnr_gain": psnr_gain,
        "ssim_gain": ssim_gain,
        "denoising_metrics": {
            "avg_psnr_noisy": avg_psnr_noisy,
            "avg_psnr_restored": avg_psnr_restored,
            "psnr_gain": psnr_gain,
            "avg_ssim_noisy": avg_ssim_noisy,
            "avg_ssim_restored": avg_ssim_restored,
            "ssim_gain": ssim_gain,
            "avg_improvement": avg_improvement
        },
        "report_written": report_written,
        "output_path": target_output_path if report_written else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASCADE Model Validation Suite")
    parser.add_argument("--data-dir", default=None, help="Directory of separated_scans")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size per organ")
    parser.add_argument("--output", default=None, help="Path to output report")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Explicitly allow overwriting existing output report file")
    args = parser.parse_args()

    run_validation(data_dir=args.data_dir, sample_size_per_organ=args.sample_size, output_path=args.output, overwrite=args.overwrite)
