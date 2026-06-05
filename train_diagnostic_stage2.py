import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────
# CONFIGURATION & LOCAL PATHS
# ────────────────────────────────────────────────────────
DATA_DIR = r"d:\Medical-Denoise-AI-CT-Scan-Restoration--main\Medical-Denoise-AI-CT-Scan-Restoration--main\data"
UNIFIED_DIR = os.path.join(DATA_DIR, "unified_dataset")
LABELS_PATH = os.path.join(UNIFIED_DIR, "labels.json")
MODEL_OUT_PATH = r"d:\Medical-Denoise-AI-CT-Scan-Restoration--main\Medical-Denoise-AI-CT-Scan-Restoration--main\medical_ct_diagnostic_model.keras"

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 15  # Backbone is frozen; trains very quickly!

# ────────────────────────────────────────────────────────
# HIGH-PERFORMANCE DATA PIPELINE
# ────────────────────────────────────────────────────────
def parse_image_and_labels(record_tensor):
    # Construct clean absolute path
    clean_abs = tf.strings.join([UNIFIED_DIR, "/", record_tensor['clean_path']])
    
    # Read grayscale CT slice
    img = tf.io.read_file(clean_abs)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to 0.0 - 1.0
    
    # Target structures for Multi-Task Outputs
    labels = {
        'class_output': record_tensor['organ_label'],
        'bbox_output': record_tensor['bbox']
    }
    return img, labels

def get_diagnostic_datasets():
    with open(LABELS_PATH) as f:
        records = json.load(f)
        
    train_records = [r for r in records if r['split'] == 'train']
    val_records = [r for r in records if r['split'] == 'val']
    
    print(f"[*] Loaded paired records for ResNet Diagnostic model [Train: {len(train_records)} | Val: {len(val_records)}]")
    
    # Create TF datasets
    def make_tf_dataset(record_list):
        clean_paths = [r['clean_path'] for r in record_list]
        organ_labels = [r['organ_label'] for r in record_list]
        bboxes = [r['bbox_normalized'] for r in record_list]
        
        # Pack target tensors safely
        ds = tf.data.Dataset.from_tensor_slices({
            'clean_path': clean_paths,
            'organ_label': organ_labels,
            'bbox': bboxes
        })
        ds = ds.map(parse_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
        return ds
        
    train_ds = make_tf_dataset(train_records).shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(val_records).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# ────────────────────────────────────────────────────────
# MULTI-TASK DEEP RESNET-50V2 DIAGNOSTIC MODEL
# ────────────────────────────────────────────────────────
def build_multi_task_diagnostic_model(input_shape=(256, 256, 1)):
    """
    Builds a Multi-Task Deep ResNet-50 Network that simultaneously:
      1. Classifies the scan organ type with high-accuracy (Chest, Brain, Abdomen)
      2. Regresses the exact bounding box of the abnormality [ymin, xmin, ymax, xmax]
    """
    inputs = layers.Input(shape=input_shape)
    
    # -- STAGE 1: Real-time Data Augmentation (active during training mode) --
    augmented = layers.RandomFlip("horizontal")(inputs)
    augmented = layers.RandomRotation(0.04)(augmented)
    augmented = layers.RandomTranslation(height_factor=0.03, width_factor=0.03)(augmented)
    
    # -- STAGE 2: Grayscale to 3-Channel Replication for ImageNet compatibility --
    x_3ch = layers.Concatenate(axis=-1)([augmented, augmented, augmented])
    
    # -- STAGE 3: Pre-trained Deep ResNet50V2 Backbone --
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # FREEZE the pre-trained backbone to preserve visual weights and accelerate training.
    base_model.trainable = False
    
    # Feature extraction
    features = base_model(x_3ch)
    gap = layers.GlobalAveragePooling2D()(features)
    
    # Dense projection block
    dense = layers.Dense(256, activation="relu")(gap)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.4)(dense)
    
    # --- Branch 1: Organ & Anomaly Classification (Chest, Brain, Abdomen) ---
    class_output = layers.Dense(3, activation="softmax", name="class_output")(dense)
    
    # --- Branch 2: Bounding Box Regression (anomaly boundaries ymin, xmin, ymax, xmax) ---
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(dense)
    
    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output], name="Diagnostic_Stage2_Model")
    
    # Compile with Adam Optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mse'
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mae'
        }
    )
    return model

# ────────────────────────────────────────────────────────
# EXECUTION FLOW
# ────────────────────────────────────────────────────────
def run_training():
    if not os.path.exists(LABELS_PATH):
        print(f"[!] Error: master labels.json not found at {LABELS_PATH}. Run prepare_unified_dataset.py first.")
        return
        
    # Load dataset pipelines
    train_ds, val_ds = get_diagnostic_datasets()
    
    # Build Model
    print("[*] Compiling Multi-Task ResNet-50V2 Diagnostic network...")
    model = build_multi_task_diagnostic_model()
    model.summary()
    
    # Train
    print(f"[*] Starting ResNet Diagnostic training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # Save
    print(f"[*] Saving trained weights to: {MODEL_OUT_PATH}")
    model.save(MODEL_OUT_PATH)
    
    # Save performance curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['class_output_accuracy'], label='Train Acc')
    plt.plot(history.history['val_class_output_accuracy'], label='Val Acc')
    plt.title('Classification Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['bbox_output_mae'], label='Train BBox MAE')
    plt.plot(history.history['val_bbox_output_mae'], label='Val BBox MAE')
    plt.title('Bounding Box Mean Absolute Error (MAE)')
    plt.legend()
    
    performance_out_path = os.path.join(DATA_DIR, "diagnostic_training_performance.png")
    plt.savefig(performance_out_path)
    print(f"[*] Performance curves successfully saved: {performance_out_path}")

if __name__ == "__main__":
    run_training()
