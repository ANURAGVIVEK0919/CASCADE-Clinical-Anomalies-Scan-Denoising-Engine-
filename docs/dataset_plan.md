# Medical Denoise AI — Dataset Problem & Combination Plan

## Version 1.0 — Current Status & Fix Roadmap

---

## 1. Abhi Kya Problem Hai

### 1.1 Current Dataset (Jo Abhi Use Ho Raha Hai)

**Dataset:** `andrewmvd/ct-low-dose-reconstruction` (Kaggle)

| Property | Value |
|---|---|
| Total scans | 40 CT scans |
| Organ coverage | **Sirf Chest** |
| Format | DICOM (.dcm) |
| Paired data | ✅ Noisy + Clean pairs (denoising ke liye theek hai) |
| Organ labels | ❌ Koi label nahi |
| Bounding box annotations | ❌ Koi annotation nahi |
| Brain scans | ❌ Nahi |
| Abdomen scans | ❌ Nahi |

### 1.2 Do Alag Models Ke Alag Requirements Hain

```
Stage 1 — U-Net Denoiser
  Zaroorat:  Noisy CT image  +  Clean CT image (paired)
  Abhi kya hai: ✅ 40 chest pairs — sirf chest ke liye kaam karta hai
  Problem:   Brain/Abdomen ke liye koi noisy/clean pair nahi

Stage 2 — ResNet-50V2 Classifier
  Zaroorat:  CT image  +  Organ label (0/1/2)  +  BBox coordinates
  Abhi kya hai: ❌ Kuch bhi nahi — na label, na bbox
  Problem:   Model ne kuch meaningful nahi seekha
             88.67% accuracy = shayad sab kuch "Chest" predict kar raha hai
```

### 1.3 Core Problem Summary

> **Ek hi dataset se denoising bhi nahi ho sakti (sirf chest) aur diagnosis bilkul nahi ho sakti (koi label nahi).** Dono models ke liye alag-alag data chahiye, jo abhi exist hi nahi karta humari pipeline mein.

---

## 2. Kya Banana Hai — Target Dataset Structure

Final combined dataset mein har ek image ke liye yeh hona chahiye:

```
dataset/
├── chest/
│   ├── noisy/          ← low-dose CT slices
│   ├── clean/          ← full-dose CT slices (paired)
│   └── labels.json     ← organ=0, bbox=[y1,x1,y2,x2]
├── brain/
│   ├── noisy/
│   ├── clean/
│   └── labels.json     ← organ=1, bbox=[y1,x1,y2,x2]
└── abdomen/
    ├── noisy/
    ├── clean/
    └── labels.json     ← organ=2, bbox=[y1,x1,y2,x2]
```

Har `labels.json` entry ka format:

```json
{
  "image_id": "brain_0042",
  "organ_label": 1,
  "bbox_normalized": [0.25, 0.30, 0.75, 0.80],
  "noisy_path": "brain/noisy/brain_0042.png",
  "clean_path": "brain/clean/brain_0042.png",
  "source_dataset": "rsna-hemorrhage",
  "split": "train"
}
```

---

## 3. Kaunse Datasets Use Karne Hain

### 3.1 Chest (Already Hai — Extend Karna Hai)

| Dataset | Kaggle Link | Kya Milega |
|---|---|---|
| **andrewmvd/ct-low-dose-reconstruction** | Abhi use ho raha hai | 40 paired chest scans |
| **RSNA Pneumonia Detection** | `rsna-pneumonia-detection-challenge` | ~26,000 chest X-ray + BBox labels |
| **SIIM-ACR Pneumothorax** | `seesound/siim-acr-pneumothorax` | Chest CT + segmentation masks |

> **Note:** RSNA Pneumonia mein BBox annotations already hain — yeh directly classifier training mein use ho sakta hai.

### 3.2 Brain (Abhi Kuch Nahi — Lana Padega)

| Dataset | Kaggle Link | Kya Milega |
|---|---|---|
| **RSNA Intracranial Hemorrhage** | `rsna-intracranial-hemorrhage-detection` | Brain CT slices + hemorrhage labels |
| **Brain Tumor MRI** | `masoudnickparvar/brain-tumor-mri-dataset` | Brain scans (MRI, CT nahi — but useful for classification) |
| **CT Brain Hemorrhage** | `felipekitamura/head-ct-hemorrhage` | 200 brain CT + normal scans with labels |

> **Recommended:** `felipekitamura/head-ct-hemorrhage` — chhota, clean, aur labels ready hain.

### 3.3 Abdomen (Abhi Kuch Nahi — Lana Padega)

| Dataset | Kaggle Link | Kya Milega |
|---|---|---|
| **LiTS Liver Tumor Segmentation** | `andrewmvd/liver-tumor-segmentation` | Abdomen CT + liver/tumor masks |
| **CT-ORG Multi-Organ** | `darren2020/ct-org-multi-organ-segmentation` | Multi-organ labeled CT slices |
| **CHAOS Abdominal CT** | `isacsurek/chaos-dataset` | Abdomen CT + organ segmentation |

> **Recommended:** `darren2020/ct-org-multi-organ-segmentation` — directly organ labels milte hain.

---

## 4. Step-by-Step Kaam Kaise Karna Hai

### Step 1 — Datasets Download Karo

```bash
# Kaggle CLI se download karo
kaggle datasets download andrewmvd/ct-low-dose-reconstruction
kaggle datasets download felipekitamura/head-ct-hemorrhage
kaggle datasets download darren2020/ct-org-multi-organ-segmentation
kaggle competitions download rsna-pneumonia-detection-challenge
```

### Step 2 — Brain/Abdomen ke liye Synthetic Noise Generate Karo

Kyunki brain/abdomen datasets mein sirf clean images hain, artificially noisy version banana padega:

```python
import numpy as np
from PIL import Image
import os

def add_ct_realistic_noise(image_array, noise_level=0.08):
    """
    CT-realistic noise: Poisson (quantum noise) + Gaussian (electronic noise)
    """
    # Normalize to [0, 1]
    img = image_array.astype(np.float32) / 255.0
    
    # Poisson noise — CT ka main noise source
    poisson = np.random.poisson(img * 1000) / 1000
    poisson_noise = poisson - img
    
    # Gaussian noise — electronic noise
    gaussian_noise = np.random.normal(0, noise_level, img.shape)
    
    # Combine
    noisy = img + poisson_noise * 0.7 + gaussian_noise * 0.3
    noisy = np.clip(noisy, 0, 1)
    
    return (noisy * 255).astype(np.uint8)

def generate_noisy_pairs(clean_dir, noisy_dir, noise_levels=[0.05, 0.08, 0.12]):
    """
    Ek clean image se multiple noise levels ke versions banao
    """
    os.makedirs(noisy_dir, exist_ok=True)
    
    for fname in os.listdir(clean_dir):
        if not fname.endswith(('.png', '.jpg')):
            continue
            
        clean_img = np.array(Image.open(os.path.join(clean_dir, fname)).convert('L'))
        
        for level in noise_levels:
            noisy_img = add_ct_realistic_noise(clean_img, noise_level=level)
            out_name = fname.replace('.png', f'_noise{int(level*100)}.png')
            Image.fromarray(noisy_img).save(os.path.join(noisy_dir, out_name))
            
    print(f"Done: {noisy_dir}")
```

### Step 3 — BBox Annotations Banana / Convert Karna

Kuch datasets mein segmentation masks hain (pixel-level), unhe bounding boxes mein convert karna padega:

```python
import numpy as np
import json

def mask_to_bbox_normalized(mask_array):
    """
    Segmentation mask se normalized bounding box nikalo
    Returns: [ymin, xmin, ymax, xmax] — sab 0-1 ke beech
    """
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    if not rows.any():
        return [0.1, 0.1, 0.9, 0.9]  # fallback
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    h, w = mask_array.shape
    return [
        round(rmin / h, 4),  # ymin
        round(cmin / w, 4),  # xmin
        round(rmax / h, 4),  # ymax
        round(cmax / w, 4)   # xmax
    ]

def build_labels_json(image_dir, mask_dir, organ_label, output_json):
    """
    Saare images ke liye labels.json banao
    """
    labels = []
    
    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith('.png'):
            continue
            
        mask_path = os.path.join(mask_dir, fname)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L')) > 127
            bbox = mask_to_bbox_normalized(mask)
        else:
            bbox = [0.15, 0.15, 0.85, 0.85]  # default bbox
        
        labels.append({
            "image_id": fname.replace('.png', ''),
            "organ_label": organ_label,
            "bbox_normalized": bbox,
            "noisy_path": f"noisy/{fname}",
            "clean_path": f"clean/{fname}"
        })
    
    with open(output_json, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Labels saved: {output_json} ({len(labels)} entries)")
```

### Step 4 — Teeno Organs Ka Data Merge Karo

```python
import json
import shutil
import os

def merge_all_datasets(chest_dir, brain_dir, abdomen_dir, output_dir):
    """
    Teeno organs ke data ko ek combined dataset mein merge karo
    """
    all_labels = []
    
    organs = [
        (chest_dir,   0, "chest"),
        (brain_dir,   1, "brain"),
        (abdomen_dir, 2, "abdomen"),
    ]
    
    for src_dir, organ_id, organ_name in organs:
        labels_path = os.path.join(src_dir, 'labels.json')
        with open(labels_path) as f:
            labels = json.load(f)
        
        for entry in labels:
            # Paths update karo
            entry['noisy_path'] = f"{organ_name}/noisy/{entry['image_id']}.png"
            entry['clean_path'] = f"{organ_name}/clean/{entry['image_id']}.png"
            entry['organ_label'] = organ_id
            all_labels.append(entry)
        
        # Files copy karo
        for folder in ['noisy', 'clean']:
            src = os.path.join(src_dir, folder)
            dst = os.path.join(output_dir, organ_name, folder)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    
    # Train/Val/Test split — 80/10/10
    np.random.shuffle(all_labels)
    n = len(all_labels)
    for i, entry in enumerate(all_labels):
        if i < n * 0.8:
            entry['split'] = 'train'
        elif i < n * 0.9:
            entry['split'] = 'val'
        else:
            entry['split'] = 'test'
    
    # Master labels file save karo
    with open(os.path.join(output_dir, 'master_labels.json'), 'w') as f:
        json.dump(all_labels, f, indent=2)
    
    print(f"Combined dataset ready: {len(all_labels)} total images")
    print(f"  Chest:   {sum(1 for l in all_labels if l['organ_label']==0)}")
    print(f"  Brain:   {sum(1 for l in all_labels if l['organ_label']==1)}")
    print(f"  Abdomen: {sum(1 for l in all_labels if l['organ_label']==2)}")
```

### Step 5 — Training Pipeline Update Karo

Naye combined dataset ke saath dono models ko retrain karna padega:

```
1. U-Net Denoiser:
   - Input:  noisy/*.png (teeno organs)
   - Target: clean/*.png (teeno organs)
   - Expected improvement: model ab sirf chest nahi, teeno organs denoise kar payega

2. ResNet-50V2 Classifier:
   - Input:   clean/*.png
   - Labels:  organ_label (0/1/2) + bbox_normalized [y1,x1,y2,x2]
   - Expected: actual 3-class classification with meaningful accuracy
```

---

## 5. Expected Dataset Size After Combining

| Source | Organ | Approx Images |
|---|---|---|
| andrewmvd/ct-low-dose-reconstruction | Chest | ~500 slices |
| rsna-pneumonia-detection-challenge | Chest | ~26,000 |
| felipekitamura/head-ct-hemorrhage | Brain | ~200 |
| darren2020/ct-org-multi-organ | Abdomen | ~1,000+ |
| **Total (estimate)** | **All 3** | **~27,000+** |

---

## 6. Priority Order — Pehle Kya Karo

```
Priority 1 (Critical):
  → felipekitamura/head-ct-hemorrhage download karo
  → darren2020/ct-org-multi-organ download karo
  → Synthetic noise generate karo brain/abdomen ke liye
  → labels.json banao teeno ke liye

Priority 2 (Important):
  → Teeno datasets merge karo (Step 4 script)
  → ResNet classifier retrain karo naye labels ke saath
  → U-Net ko brain/abdomen data pe bhi fine-tune karo

Priority 3 (Nice to have):
  → RSNA Pneumonia data add karo chest ke liye (volume badhao)
  → Data augmentation (flip, rotate, brightness)
  → Cross-validation setup
```

---

## 7. Quick Checklist

- [x] NIH DeepLesion metadata parsed successfully
- [x] Brain scans selected from active data folders
- [x] Abdomen scans selected from active data folders
- [x] Brain synthetic noisy pairs generated (Poisson + Gaussian)
- [x] Abdomen synthetic noisy pairs generated (Poisson + Gaussian)
- [x] Chest synthetic noisy pairs generated (Poisson + Gaussian)
- [x] Unified multi-organ master `labels.json` compiled with normalized BBoxes and organ classes
- [ ] U-Net retrained on all three organs (Chest, Brain, Abdomen)
- [ ] ResNet-50V2 retrained on the unified dataset
- [ ] Real-world multi-organ diagnostic and denoising verified in browser portal

---

*Document version: 1.0 | Medical Denoise AI Project*