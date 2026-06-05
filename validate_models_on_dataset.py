"""
Comprehensive validation script to run inference on all dataset images.
Tests:
1. Denoising performance (PSNR/SSIM improvements)
2. Diagnostic organ classification accuracy (Chest vs Brain vs Abdomen)
3. Diagnostic severity prediction distributions
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time

# Ensure we can import app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_image, get_model, get_diagnostic_model, run_diagnostic_analyzer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEPARATED_DIR = os.path.join(DATA_DIR, "separated_scans")

def run_validation():
    print("=" * 65)
    print("  MED-DENOISE AI: DATASET-WIDE MODEL VALIDATION SUITE")
    print("=" * 65)
    
    # Pre-load models
    model = get_model()
    diag_model = get_diagnostic_model()
    
    if model is None:
        print("[!] Error: Denoising U-Net model could not be loaded!")
        return
    if diag_model is None:
        print("[!] Error: Diagnostic ResNet model could not be loaded!")
        return
        
    print("[+] Both models loaded successfully into memory.")
    
    organs = ["chest", "brain", "abdomen"]
    folders = ["normal", "medium_risk", "high_risk"]
    
    total_images = 0
    correct_organ_preds = 0
    
    # Accumulators for denoising metrics
    psnr_noisy_sum = 0
    psnr_restored_sum = 0
    ssim_noisy_sum = 0
    ssim_restored_sum = 0
    improvement_sum = 0
    
    # Severity distribution: actual_folder -> predicted_severity -> count
    severity_dist = {
        "normal": {"Normal": 0, "Mild Anomaly": 0, "High Risk": 0},
        "medium_risk": {"Normal": 0, "Mild Anomaly": 0, "High Risk": 0},
        "high_risk": {"Normal": 0, "Mild Anomaly": 0, "High Risk": 0}
    }
    
    # Organ prediction counts: actual_organ -> predicted_organ -> count
    organ_matrix = {o: {pred_o: 0 for pred_o in organs} for o in organs}
    
    # Collect all noisy file paths (balanced sampling: 50 per organ)
    import random
    random.seed(42)
    
    file_list = []
    for organ in organs:
        organ_files = []
        for folder in folders:
            noisy_dir = os.path.join(SEPARATED_DIR, organ, folder, "noisy")
            if not os.path.exists(noisy_dir):
                continue
            files = [f for f in os.listdir(noisy_dir) if f.endswith('.png')]
            for f in files:
                organ_files.append({
                    'file_path': os.path.join(noisy_dir, f),
                    'filename': f,
                    'actual_organ': organ,
                    'actual_severity': folder
                })
        
        # Sample up to 50 files for this organ
        sample_size = min(50, len(organ_files))
        sampled = random.sample(organ_files, sample_size)
        file_list.extend(sampled)
        print(f"[*] Sampled {sample_size} scans for organ: {organ.upper()} (out of {len(organ_files)})")
                
    total_files = len(file_list)
    print(f"[*] Total sampled scans to validate: {total_files}")
    if total_files == 0:
        print("[!] No images found to validate.")
        return
        
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
        class_idx = np.argmax(class_pred[0])
        class_map = {0: 'chest', 1: 'brain', 2: 'abdomen'}
        pred_organ = class_map.get(class_idx, 'chest')
        
        # Update organ classification metrics
        organ_matrix[actual_organ][pred_organ] += 1
        if pred_organ == actual_organ:
            correct_organ_preds += 1
            
        # 3. Diagnostic Analyzer (Severity Risk Levels)
        ymin, xmin, ymax, xmax = bbox_pred[0].tolist()
        bbox = [ymin, xmin, ymax, xmax]
        diagnosis = run_diagnostic_analyzer(denoised_arr, actual_organ, bbox, pred_organ)
        pred_severity = diagnosis['severity']
        
        severity_dist[actual_severity][pred_severity] += 1
        
        # Update denoising metrics
        psnr_noisy_sum += metrics['psnr_noisy']
        psnr_restored_sum += metrics['psnr_restored']
        ssim_noisy_sum += metrics['ssim_noisy']
        ssim_restored_sum += metrics['ssim_restored']
        improvement_sum += metrics['improvement']
        
        total_images += 1
        
        # Print progress every 100 images
        if total_images % 100 == 0 or total_images == total_files:
            elapsed = time.time() - start_time
            img_per_sec = total_images / elapsed
            print(f"    Processed {total_images}/{total_files} | Organ Accuracy: {100.0 * correct_organ_preds / total_images:.1f}% | Speed: {img_per_sec:.1f} img/sec")
            
    total_time = time.time() - start_time
    
    # Calculate Averages
    avg_psnr_noisy = psnr_noisy_sum / total_images
    avg_psnr_restored = psnr_restored_sum / total_images
    avg_ssim_noisy = ssim_noisy_sum / total_images
    avg_ssim_restored = ssim_restored_sum / total_images
    avg_improvement = improvement_sum / total_images
    organ_accuracy = (correct_organ_preds / total_images) * 100.0
    
    # Write report
    report_path = os.path.join(DATA_DIR, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write("==========================================================\n")
        f.write("      MEDICAL CT RESTORATION & DIAGNOSTICS REPORT\n")
        f.write("==========================================================\n")
        f.write(f"Total Scans Processed: {total_images}\n")
        f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        f.write(f"Average Processing Speed: {total_images / total_time:.2f} images/sec\n\n")
        
        f.write("----------------------------------------------------------\n")
        f.write("1. DENOISING PERFORMANCE METRICS\n")
        f.write("----------------------------------------------------------\n")
        f.write(f"Average PSNR (Noisy Baseline): {avg_psnr_noisy:.2f} dB\n")
        f.write(f"Average PSNR (Denoised U-Net):  {avg_psnr_restored:.2f} dB\n")
        f.write(f"Average PSNR Gain:              +{avg_psnr_restored - avg_psnr_noisy:.2f} dB\n\n")
        
        f.write(f"Average SSIM (Noisy Baseline): {avg_ssim_noisy:.4f}\n")
        f.write(f"Average SSIM (Denoised U-Net):  {avg_ssim_restored:.4f}\n")
        f.write(f"Average SSIM Gain:              +{avg_ssim_restored - avg_ssim_noisy:.4f}\n\n")
        f.write(f"Average Signal Quality Improvement: {avg_improvement:.2f}%\n\n")
        
        f.write("----------------------------------------------------------\n")
        f.write("2. ORGAN CLASSIFICATION ACCURACY\n")
        f.write("----------------------------------------------------------\n")
        f.write(f"Overall Classification Accuracy: {organ_accuracy:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write("Actual \\ Predicted |   Chest  |   Brain  |  Abdomen \n")
        f.write("-------------------|----------|----------|----------\n")
        for actual in organs:
            f.write(f"{actual.capitalize():<18} | ")
            for pred in organs:
                f.write(f"{organ_matrix[actual][pred]:<8} | ")
            f.write("\n")
        f.write("\n")
        
        f.write("----------------------------------------------------------\n")
        f.write("3. DIAGNOSTIC SEVERITY ASSESSMENT\n")
        f.write("----------------------------------------------------------\n")
        f.write("Validation of Risk Severity Mapping:\n")
        for actual_sev in ["normal", "medium_risk", "high_risk"]:
            f.write(f"Actual Category '{actual_sev.upper()}':\n")
            for pred_sev in ["Normal", "Mild Anomaly", "High Risk"]:
                count = severity_dist[actual_sev][pred_sev]
                pct = (count / max(1, sum(severity_dist[actual_sev].values()))) * 100.0
                f.write(f"  -> Predicted {pred_sev:<12}: {count:<5} ({pct:.1f}%)\n")
            f.write("\n")
            
    print("\n" + "=" * 65)
    print("  VALIDATION COMPLETE! Summary Report:")
    print("=" * 65)
    print(f"  -> Total Processed: {total_images} scans")
    print(f"  -> Denoising Quality Gain: +{avg_psnr_restored - avg_psnr_noisy:.2f} dB PSNR")
    print(f"  -> Denoising Structural Gain: +{avg_ssim_restored - avg_ssim_noisy:.4f} SSIM")
    print(f"  -> Signal Quality Improvement: {avg_improvement:.1f}%")
    print(f"  -> Organ Classification Accuracy: {organ_accuracy:.2f}%")
    print("-" * 65)
    print(f"  Full report saved to: {report_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_validation()
