import os
import pandas as pd
import numpy as np
import json
import shutil
from PIL import Image

# ────────────────────────────────────────────────────────
# CONFIGURATION & DIRECTORIES
# ────────────────────────────────────────────────────────
DATA_DIR = r"d:\Medical-Denoise-AI-CT-Scan-Restoration--main\Medical-Denoise-AI-CT-Scan-Restoration--main\data"
CSV_PATH = os.path.join(DATA_DIR, "DL_info.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "unified_dataset")

CLEAN_OUT_DIR = os.path.join(OUTPUT_DIR, "clean")
NOISY_OUT_DIR = os.path.join(OUTPUT_DIR, "noisy")

os.makedirs(CLEAN_OUT_DIR, exist_ok=True)
os.makedirs(NOISY_OUT_DIR, exist_ok=True)

# Dataset limits for balancing
LIMIT_PER_ORGAN = 3000 # 3000 Chest, 3000 Brain, 3000 Abdomen = 9000 total paired images!
NOISE_LEVEL = 0.08    # Standard electronic noise factor

# ────────────────────────────────────────────────────────
# NOISE GENERATOR (POISSON + GAUSSIAN DUAL SCATTERING)
# ────────────────────────────────────────────────────────
def add_ct_realistic_noise(image_array, noise_level=0.08):
    """
    Simulates high-fidelity clinical low-dose CT noise:
    - Poisson Noise (Quantum noise representing X-ray photon count fluctuations)
    - Gaussian Noise (Electronic thermal noise representing scanner detector variance)
    """
    # Normalize to [0.0, 1.0]
    img = image_array.astype(np.float32) / 255.0
    
    # Poisson noise (main CT scatter variance)
    # Scale to simulate high-intensity X-ray counts
    scaled = np.clip(img * 1000.0, 0.0, 1000.0)
    poisson = np.random.poisson(scaled) / 1000.0
    poisson_noise = poisson - img
    
    # Gaussian noise (thermal detector electronic variance)
    gaussian_noise = np.random.normal(0, noise_level, img.shape)
    
    # Combine (70% quantum mottle + 30% electronic noise)
    noisy = img + poisson_noise * 0.7 + gaussian_noise * 0.3
    noisy = np.clip(noisy, 0.0, 1.0)
    
    return (noisy * 255.0).astype(np.uint8)

# ────────────────────────────────────────────────────────
# METADATA PARSER & MATCHER
# ────────────────────────────────────────────────────────
def parse_and_generate_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: NIH DeepLesion metadata not found at: {CSV_PATH}")
        return
        
    print(f"[*] Parsing DeepLesion metadata from: {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Counts tracker
    counts = {0: 0, 1: 0, 2: 0} # 0: Chest, 1: Brain, 2: Abdomen
    dataset_records = []
    
    print("[*] Filtering, scaling, and copying paired slices...")
    
    for idx, row in df.iterrows():
        file_name = row['File_name']
        bbox_str = row['Bounding_boxes']
        organ_class = int(row['Coarse_lesion_type'])
        
        if pd.isna(bbox_str) or pd.isna(organ_class):
            continue
            
        # Map NIH classes to simplified organ categories
        simplified_class = -1
        if organ_class in [3, 6]:
            simplified_class = 0  # Chest CT
        elif organ_class in [8]:
            simplified_class = 1  # Brain CT
        elif organ_class in [2, 4, 7]:
            simplified_class = 2  # Abdomen CT
            
        if simplified_class == -1:
            continue
            
        # Skip if organ category has reached limit
        if counts[simplified_class] >= LIMIT_PER_ORGAN:
            # Check if all classes are filled to stop early
            if all(c >= LIMIT_PER_ORGAN for c in counts.values()):
                break
            continue
            
        try:
            # Coordinates in NIH DeepLesion: xmin, ymin, xmax, ymax (size is 512x512)
            coords = [float(c) for c in bbox_str.split(',')]
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
            
            # Normalize to 0.0 - 1.0
            norm_ymin = max(0.0, min(1.0, ymin / 512.0))
            norm_xmin = max(0.0, min(1.0, xmin / 512.0))
            norm_ymax = max(0.0, min(1.0, ymax / 512.0))
            norm_xmax = max(0.0, min(1.0, xmax / 512.0))
            
            # Map filenames to folder tree path (e.g. 000001_01_01_109.png -> 000001_01_01/109.png)
            last_underscore_idx = file_name.rfind('_')
            if last_underscore_idx != -1:
                subdir = file_name[:last_underscore_idx]
                actual_file = file_name[last_underscore_idx+1:]
                rel_path = os.path.join(subdir, actual_file)
            else:
                rel_path = file_name
                
            # Check which Images_png_XX folder contains this slice
            found_path = None
            search_folders = [f"Images_png_{i:02d}" for i in range(1, 16)] # DeepLesion 1 to 15
            for folder_name in search_folders:
                check_path = os.path.join(DATA_DIR, folder_name, "Images_png", rel_path)
                if os.path.exists(check_path):
                    found_path = check_path
                    break
                    
            if found_path is None:
                continue
                
            # Unique file code for export name
            file_code = file_name.replace('.png', '')
            out_clean_name = f"{file_code}_clean.png"
            out_noisy_name = f"{file_code}_noisy.png"
            
            out_clean_path = os.path.join(CLEAN_OUT_DIR, out_clean_name)
            out_noisy_path = os.path.join(NOISY_OUT_DIR, out_noisy_name)
            
            # 1. Load clean image and scale 16-bit to 8-bit correctly
            img_raw = Image.open(found_path)
            arr_raw = np.array(img_raw).astype(np.float32)
            arr_min = np.min(arr_raw)
            arr_max = np.max(arr_raw)
            if arr_max - arr_min > 0:
                arr_scaled = (arr_raw - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr_scaled = np.zeros_like(arr_raw)
            
            img_8bit = Image.fromarray(arr_scaled.astype(np.uint8))
            img_resized = img_8bit.resize((256, 256), Image.Resampling.BILINEAR)
            img_array = np.array(img_resized)
            
            # 2. Add realistic quantum CT noise
            noisy_array = add_ct_realistic_noise(img_array, noise_level=NOISE_LEVEL)
            
            # 3. Save both files
            Image.fromarray(img_array).save(out_clean_path)
            Image.fromarray(noisy_array).save(out_noisy_path)
            
            # 4. Save metadata record
            dataset_records.append({
                "image_id": file_code,
                "organ_label": simplified_class,
                "bbox_normalized": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
                "clean_path": f"clean/{out_clean_name}",
                "noisy_path": f"noisy/{out_noisy_name}"
            })
            
            counts[simplified_class] += 1
            
            # Status reporting
            total_extracted = sum(counts.values())
            if total_extracted % 100 == 0:
                print(f" -> Processed {total_extracted} slices [Chest: {counts[0]}/{LIMIT_PER_ORGAN} | Brain: {counts[1]}/{LIMIT_PER_ORGAN} | Abdomen: {counts[2]}/{LIMIT_PER_ORGAN}]")
                
        except Exception as e:
            # Silently skip errors to continue processing pipeline
            continue
            
    # Train / Validation Split (85% Train, 15% Validation)
    np.random.seed(42)
    np.random.shuffle(dataset_records)
    n = len(dataset_records)
    train_size = int(n * 0.85)
    
    for i, entry in enumerate(dataset_records):
        entry['split'] = 'train' if i < train_size else 'val'
        
    # Write Master labels.json
    labels_out_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(labels_out_path, 'w') as f:
        json.dump(dataset_records, f, indent=2)
        
    print("  [SUCCESS] UNIFIED MEDICAL DIAGNOSTIC & Restorations DATASET READY!")
    print("="*80)
    print(f"[*] Total Extracted Slices: {len(dataset_records)}")
    print(f"  - Chest Slices (Class 0):   {counts[0]}")
    print(f"  - Brain Slices (Class 1):   {counts[1]}")
    print(f"  - Abdomen Slices (Class 2): {counts[2]}")
    print(f"[*] Master Annotations File: {labels_out_path}")
    print(f"[*] Slices Clean Folder:      {CLEAN_OUT_DIR}")
    print(f"[*] Slices Noisy Folder:      {NOISY_OUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    parse_and_generate_dataset()
