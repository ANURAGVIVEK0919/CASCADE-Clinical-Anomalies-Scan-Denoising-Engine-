#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Medical CT Scan AI — Kaggle GPU Training Pipeline          ║
║  Trains both U-Net Denoiser + ResNet50V2 Diagnostic Model   ║
╚══════════════════════════════════════════════════════════════╝

Upload Instructions:
1. Upload 'data.zip' as a Kaggle Dataset (name it 'medical-ct-data')
2. Create a new Kaggle Notebook with GPU accelerator enabled
3. Add the dataset to the notebook
4. Paste this script into a code cell and run
5. After training, download the .keras files from /kaggle/working/
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("  Medical CT Scan AI — Kaggle GPU Training Pipeline")
print("=" * 60)
print(f"  TensorFlow version: {tf.__version__}")
print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
for gpu in tf.config.list_physical_devices('GPU'):
    print(f"    → {gpu}")
print("=" * 60)

# ────────────────────────────────────────────────────────
# AUTO-DETECT DATA PATHS
# ────────────────────────────────────────────────────────
DATA_DIR = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working"

# Find separated_scans directory (handles any dataset name)
sep_scan_matches = glob.glob(os.path.join(DATA_DIR, "**/separated_scans"), recursive=True)
unified_matches = glob.glob(os.path.join(DATA_DIR, "**/unified_dataset"), recursive=True)

if not sep_scan_matches:
    raise FileNotFoundError(
        "Could not find 'separated_scans' folder. "
        "Make sure you uploaded data.zip as a Kaggle Dataset and added it to this notebook."
    )

separated_dir = sep_scan_matches[0]
unified_dir = unified_matches[0] if unified_matches else None
labels_path = os.path.join(unified_dir, "labels.json") if unified_dir else None

print(f"\n[*] Separated scans: {separated_dir}")
print(f"[*] Unified dataset: {unified_dir}")
print(f"[*] Labels file: {labels_path}")

IMG_SIZE = (256, 256)
BATCH_SIZE = 32  # Increase if GPU memory allows (try 64 on T4/P100)


# ════════════════════════════════════════════════════════════
# MODEL 1: HIGH-FIDELITY U-NET DENOISER
# ════════════════════════════════════════════════════════════

def parse_image_pair(noisy_path, clean_path):
    """Load and preprocess a noisy/clean image pair."""
    noisy = tf.io.read_file(noisy_path)
    noisy = tf.image.decode_png(noisy, channels=1)
    noisy = tf.image.resize(noisy, IMG_SIZE)
    noisy = tf.cast(noisy, tf.float32) / 255.0

    clean = tf.io.read_file(clean_path)
    clean = tf.image.decode_png(clean, channels=1)
    clean = tf.image.resize(clean, IMG_SIZE)
    clean = tf.cast(clean, tf.float32) / 255.0

    return noisy, clean


def collect_denoising_pairs():
    """Collect all noisy/clean image pairs from separated_scans."""
    noisy_paths, clean_paths = [], []
    organs = ["chest", "brain", "abdomen"]
    severities = ["normal", "medium_risk", "high_risk"]

    for organ in organs:
        for severity in severities:
            noisy_dir = os.path.join(separated_dir, organ, severity, "noisy")
            clear_dir = os.path.join(separated_dir, organ, severity, "clear")

            if not os.path.isdir(noisy_dir) or not os.path.isdir(clear_dir):
                continue

            files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
            matched = 0
            for f in files:
                clean_f = f.replace('_noisy', '_clean')
                clean_path = os.path.join(clear_dir, clean_f)
                if os.path.exists(clean_path):
                    noisy_paths.append(os.path.join(noisy_dir, f))
                    clean_paths.append(clean_path)
                    matched += 1
            print(f"    {organ}/{severity}: {matched} pairs")

    return noisy_paths, clean_paths


def build_unet(input_shape=(256, 256, 1)):
    """Build a U-Net with Batch Normalization for CT denoising."""
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = conv_block(p3, 256)

    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = conv_block(u5, 128)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = conv_block(u6, 64)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = conv_block(u7, 32)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet_CT_Denoiser")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_unet():
    """Train the U-Net denoiser model."""
    print("\n" + "=" * 60)
    print("  PART 1: Training U-Net CT Denoiser")
    print("=" * 60)

    noisy_paths, clean_paths = collect_denoising_pairs()
    num_samples = len(noisy_paths)
    print(f"\n[*] Total paired images: {num_samples}")

    if num_samples == 0:
        print("[!] No paired images found — skipping U-Net training.")
        return None

    # Train/Val split
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_split = int(0.1 * num_samples)

    train_noisy = [noisy_paths[i] for i in indices[val_split:]]
    train_clean = [clean_paths[i] for i in indices[val_split:]]
    val_noisy = [noisy_paths[i] for i in indices[:val_split]]
    val_clean = [clean_paths[i] for i in indices[:val_split]]

    print(f"[*] Train: {len(train_noisy)} | Val: {len(val_noisy)}")

    train_ds = (tf.data.Dataset.from_tensor_slices((train_noisy, train_clean))
                .map(parse_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((val_noisy, val_clean))
              .map(parse_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model = build_unet()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "medical_ct_denoiser_full.keras"),
            monitor='val_loss', save_best_only=True, mode='min', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

    # Save final model
    save_path = os.path.join(OUTPUT_DIR, "medical_ct_denoiser_full.keras")
    model.save(save_path)
    print(f"[+] U-Net saved: {save_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('U-Net Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('U-Net MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "unet_training_curves.png"), dpi=150)
    plt.show()

    return model


# ════════════════════════════════════════════════════════════
# MODEL 2: MULTI-TASK RESNET50V2 DIAGNOSTIC MODEL
# ════════════════════════════════════════════════════════════

def parse_diagnostic_record(record):
    """Load and preprocess a diagnostic record."""
    clean_abs = tf.strings.join([unified_dir, "/", record['clean_path']])
    img = tf.io.read_file(clean_abs)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    labels = {
        'class_output': record['organ_label'],
        'bbox_output': record['bbox']
    }
    return img, labels


def build_diagnostic_model(input_shape=(256, 256, 1)):
    """Build a multi-task ResNet50V2 for organ classification + bbox regression."""
    inputs = layers.Input(shape=input_shape)

    # Real-time augmentation (Rotation-Invariant & Sharp)
    augmented = layers.RandomFlip("horizontal_and_vertical")(inputs)
    
    # Precise 90-degree rotations to prevent boundary interpolation blurring
    def random_rot90(x):
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        return tf.image.rot90(x, k)
        
    augmented = layers.Lambda(random_rot90)(augmented)
    augmented = layers.RandomTranslation(height_factor=0.03, width_factor=0.03)(augmented)

    # Grayscale → 3-channel
    x_3ch = layers.Concatenate(axis=-1)([augmented, augmented, augmented])

    # Pre-trained ResNet50V2 backbone
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone initially

    features = base_model(x_3ch)
    gap = layers.GlobalAveragePooling2D()(features)

    dense = layers.Dense(256, activation="relu")(gap)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.4)(dense)

    # Multi-task heads
    class_output = layers.Dense(3, activation="softmax", name="class_output")(dense)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(dense)

    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output],
                         name="Diagnostic_ResNet50V2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mse'
        },
        loss_weights={'class_output': 1.0, 'bbox_output': 0.5},
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mae'
        }
    )
    return model


def train_diagnostic():
    """Train the diagnostic multi-task model."""
    print("\n" + "=" * 60)
    print("  PART 2: Training Multi-Task Diagnostic Model")
    print("=" * 60)

    if labels_path is None or not os.path.exists(labels_path):
        print("[!] No labels.json found — skipping diagnostic training.")
        return None

    with open(labels_path) as f:
        records = json.load(f)

    train_records = [r for r in records if r['split'] == 'train']
    val_records = [r for r in records if r['split'] == 'val']
    print(f"[*] Diagnostic records: Train={len(train_records)} | Val={len(val_records)}")

    def make_ds(record_list):
        ds = tf.data.Dataset.from_tensor_slices({
            'clean_path': [r['clean_path'] for r in record_list],
            'organ_label': [r['organ_label'] for r in record_list],
            'bbox': [r['bbox_normalized'] for r in record_list]
        })
        ds = ds.map(parse_diagnostic_record, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_records).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = make_ds(val_records).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_diagnostic_model()
    model.summary()

    # Stage 1: Frozen backbone
    print("\n[*] Stage 1: Training with frozen ResNet50V2 backbone...")
    callbacks_s1 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "medical_ct_diagnostic_model.keras"),
            monitor='val_loss', save_best_only=True, mode='min', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
    ]

    history1 = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks_s1)

    # Stage 2: Fine-tune top layers of backbone
    print("\n[*] Stage 2: Fine-tuning top backbone layers...")
    base = model.get_layer("resnet50v2")
    base.trainable = True
    # Freeze all but top ~80 layers to adapt deeper representation filters to grayscale CT scan textures
    for layer in base.layers[:-80]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mse'
        },
        loss_weights={'class_output': 1.0, 'bbox_output': 0.5},
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mae'
        }
    )

    callbacks_s2 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "medical_ct_diagnostic_model.keras"),
            monitor='val_loss', save_best_only=True, mode='min', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
        )
    ]

    history2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks_s2)

    # Save final model
    save_path = os.path.join(OUTPUT_DIR, "medical_ct_diagnostic_model.keras")
    model.save(save_path)
    print(f"[+] Diagnostic model saved: {save_path}")

    # Plot combined training curves
    all_loss = history1.history['loss'] + history2.history['loss']
    all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
    all_acc = history1.history['class_output_accuracy'] + history2.history['class_output_accuracy']
    all_val_acc = history1.history['val_class_output_accuracy'] + history2.history['val_class_output_accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(all_loss) + 1)
    stage1_end = len(history1.history['loss'])

    axes[0].plot(epochs_range, all_loss, label='Train Loss')
    axes[0].plot(epochs_range, all_val_loss, label='Val Loss')
    axes[0].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Fine-tune start')
    axes[0].set_title('Diagnostic Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, all_acc, label='Train Acc')
    axes[1].plot(epochs_range, all_val_acc, label='Val Acc')
    axes[1].axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Fine-tune start')
    axes[1].set_title('Diagnostic Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_training_curves.png"), dpi=150)
    plt.show()

    return model


# ════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Train both models
    unet = train_unet()
    diag = train_diagnostic()

    print("\n" + "=" * 60)
    print("  ✅ ALL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  Output files in {OUTPUT_DIR}:")
    for f in os.listdir(OUTPUT_DIR):
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 * 1024)
        print(f"    📄 {f} ({size_mb:.1f} MB)")
    print("\n  Download the .keras files and place them in your project root.")
    print("=" * 60)
