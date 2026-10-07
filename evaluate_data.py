import os
import glob
import numpy as np
import tensorflow as tf
from app import get_model, get_diagnostic_model, process_image, run_diagnostic_analyzer

def evaluate_directory(data_dir):
    print("Loading models...")
    get_model()
    m_diag = get_diagnostic_model()
    
    if m_diag is None:
        print("ERROR: Diagnostic model failed to load!")
        return
        
    print("Models loaded successfully.")
    
    results = {
        'high_risk': {'total': 0, 'correct': 0, 'bbox_areas': []},
        'normal': {'total': 0, 'correct': 0, 'bbox_areas': []}
    }
    
    # We only test the noisy images because that's what the pipeline takes as input
    search_pattern = os.path.join(data_dir, "**", "noisy", "*.png")
    all_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(all_files)} images to analyze in {data_dir}...")
    
    # Limit to 50 for a quick evaluation, otherwise it will take too long
    if len(all_files) > 50:
        import random
        random.shuffle(all_files)
        all_files = all_files[:50]
        print(f"Limiting to 50 random images for quick evaluation.")
        
    for i, file_path in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}] Processing {os.path.basename(file_path)}...")
        
        # Ground truth
        is_high_risk = 'high_risk' in file_path.lower()
        is_normal = 'normal' in file_path.lower()
        
        if not is_high_risk and not is_normal:
            continue
            
        organ = 'chest'
        if 'brain' in file_path.lower():
            organ = 'brain'
        elif 'abdomen' in file_path.lower():
            organ = 'abdomen'
            
        # Run process_image
        noisy, denoised, diff, metrics, meta = process_image(file_path)
        
        # Run diagnostic inference
        input_tensor = np.expand_dims(denoised, axis=0)
        class_pred, bbox_pred = m_diag.predict(input_tensor, verbose=0)
        
        ymin, xmin, ymax, xmax = bbox_pred[0].tolist()
        bbox = [ymin, xmin, ymax, xmax]
        
        # Run our analyzer
        diagnosis = run_diagnostic_analyzer(denoised, organ, bbox, predicted_organ=organ, filename="test.png")
        severity = diagnosis['severity']
        
        bbox_area = max(0.0, ymax - ymin) * max(0.0, xmax - xmin)
        
        if is_high_risk:
            results['high_risk']['total'] += 1
            results['high_risk']['bbox_areas'].append(bbox_area)
            if severity == "High Risk":
                results['high_risk']['correct'] += 1
        elif is_normal:
            results['normal']['total'] += 1
            results['normal']['bbox_areas'].append(bbox_area)
            if severity == "Normal":
                results['normal']['correct'] += 1
                
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for category in ['high_risk', 'normal']:
        stats = results[category]
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"Category: {category.upper()}")
            print(f"Accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
            print("-" * 30)

if __name__ == "__main__":
    evaluate_directory(r"d:\DATA")
