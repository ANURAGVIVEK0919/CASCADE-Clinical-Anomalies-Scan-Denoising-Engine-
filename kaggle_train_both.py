import os
import glob
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────
# Kaggle input path mapping (will find data zip output)
DATA_DIR = "/kaggle/input"
base_data_folder = glob.glob(os.path.join(DATA_DIR, "**/data"), recursive=True)[0]
print(f"[*] Found data folder at: {base_data_folder}")

separated_dir = os.path.join(base_data_folder, "separated_scans")
unified_dir = os.path.join(base_data_folder, "unified_dataset")
labels_path = os.path.join(unified_dir, "labels.json")

IMG_SIZE = (256, 256)

# ────────────────────────────────────────────────────────
# MODEL 1: HIGH-FIDELITY U-NET DENOISER WITH BATCH NORMALIZATION
# ────────────────────────────────────────────────────────
def parse_image_pairs_direct(noisy_abs, clean_abs):
    noisy_img = tf.io.read_file(noisy_abs)
    noisy_img = tf.image.decode_png(noisy_img, channels=1)
    noisy_img = tf.image.resize(noisy_img, IMG_SIZE)
    noisy_img = tf.cast(noisy_img, tf.float32) / 255.0
    
    clean_img = tf.io.read_file(clean_abs)
    clean_img = tf.image.decode_png(clean_img, channels=1)
    clean_img = tf.image.resize(clean_img, IMG_SIZE)
    clean_img = tf.cast(clean_img, tf.float32) / 255.0
    
    return noisy_img, clean_img

def get_denoising_datasets():
    noisy_paths = []
    clean_paths = []
    
    organs = ["chest", "brain", "abdomen"]
    severities = ["normal", "medium_risk", "high_risk"]
    
    for organ in organs:
        for severity in severities:
            noisy_dir = os.path.join(separated_dir, organ, severity, "noisy")
            clear_dir = os.path.join(separated_dir, organ, severity, "clear")
            
            if not os.path.exists(noisy_dir) or not os.path.exists(clear_dir):
                continue
                
            files = [f for f in os.listdir(noisy_dir) if f.endswith('.png')]
            for f in files:
                noisy_path = os.path.join(noisy_dir, f)
                clean_f = f.replace('_noisy', '_clean')
                clean_path = os.path.join(clear_dir, clean_f)
                
                if os.path.exists(clean_path):
                    noisy_paths.append(noisy_path)
                    clean_paths.append(clean_path)
                    
    print(f"[*] Loaded {len(noisy_paths)} paired images from augmented dataset.")
    
    num_samples = len(noisy_paths)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_split = int(0.1 * num_samples)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]
    
    train_noisy = [noisy_paths[i] for i in train_indices]
    train_clean = [clean_paths[i] for i in train_indices]
    
    val_noisy = [noisy_paths[i] for i in val_indices]
    val_clean = [clean_paths[i] for i in val_indices]
    
    def make_tf_dataset(noisy_list, clean_list):
        ds = tf.data.Dataset.from_tensor_slices((noisy_list, clean_list))
        ds = ds.map(parse_image_pairs_direct, num_parallel_calls=tf.data.AUTOTUNE)
        return ds
        
    train_ds = make_tf_dataset(train_noisy, train_clean).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(val_noisy, val_clean).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def build_unet(input_shape=(256, 256, 1)):
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
        
    inputs = layers.Input(input_shape)
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 128)
    
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = conv_block(u4, 64)
    
    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = conv_block(u5, 32)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet_CT_Denoiser")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

# ────────────────────────────────────────────────────────
# MODEL 2: MULTI-TASK DEEP RESNET-50V2 DIAGNOSTIC MODEL
# ────────────────────────────────────────────────────────
def parse_image_and_labels(record_tensor):
    clean_abs = tf.strings.join([unified_dir, "/", record_tensor['clean_path']])
    img = tf.io.read_file(clean_abs)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    
    labels = {
        'class_output': record_tensor['organ_label'],
        'bbox_output': record_tensor['bbox']
    }
    return img, labels

def get_diagnostic_datasets():
    with open(labels_path) as f:
        records = json.load(f)
        
    train_records = [r for r in records if r['split'] == 'train']
    val_records = [r for r in records if r['split'] == 'val']
    
    print(f"[*] Diagnostic records: [Train: {len(train_records)} | Val: {len(val_records)}]")
    
    def make_tf_dataset(record_list):
        clean_paths = [r['clean_path'] for r in record_list]
        organ_labels = [r['organ_label'] for r in record_list]
        bboxes = [r['bbox_normalized'] for r in record_list]
        
        ds = tf.data.Dataset.from_tensor_slices({
            'clean_path': clean_paths,
            'organ_label': organ_labels,
            'bbox': bboxes
        })
        ds = ds.map(parse_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
        return ds
        
    train_ds = make_tf_dataset(train_records).shuffle(300).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(val_records).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def build_multi_task_diagnostic_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Real-time Data Augmentation
    augmented = layers.RandomFlip("horizontal")(inputs)
    augmented = layers.RandomRotation(0.04)(augmented)
    augmented = layers.RandomTranslation(height_factor=0.03, width_factor=0.03)(augmented)
    
    # Grayscale to 3-Channel Replication
    x_3ch = layers.Concatenate(axis=-1)([augmented, augmented, augmented])
    
    # Pre-trained ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    features = base_model(x_3ch)
    gap = layers.GlobalAveragePooling2D()(features)
    
    dense = layers.Dense(256, activation="relu")(gap)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.4)(dense)
    
    class_output = layers.Dense(3, activation="softmax", name="class_output")(dense)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(dense)
    
    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output], name="Diagnostic_Stage2_Model")
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
# MAIN PIPELINE EXECUTION
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n==========================================================")
    print("[*] STARTING KAGGLE GPU DUAL MODEL TRAINING PIPELINE")
    print("==========================================================\n")
    
    # --- Part 1: U-Net Denoiser Training ---
    print("\n--- [PART 1] Training U-Net Denoiser ---")
    train_ds, val_ds = get_denoising_datasets()
    unet_model = build_unet()
    
    unet_callbacks = [
        tf.keras.callbacks.ModelCheckpoint("medical_ct_denoiser_full.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    unet_model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=unet_callbacks)
    unet_model.save("medical_ct_denoiser_full.keras")
    print("[+] Model 1 saved: medical_ct_denoiser_full.keras")
    
    # --- Part 2: ResNet Diagnostic Training ---
    print("\n--- [PART 2] Training Multi-Task Diagnostic Model ---")
    diag_train_ds, diag_val_ds = get_diagnostic_datasets()
    diag_model = build_multi_task_diagnostic_model()
    
    diag_callbacks = [
        tf.keras.callbacks.ModelCheckpoint("medical_ct_diagnostic_model.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    ]
    
    diag_model.fit(diag_train_ds, validation_data=diag_val_ds, epochs=15, callbacks=diag_callbacks)
    diag_model.save("medical_ct_diagnostic_model.keras")
    print("[+] Model 2 saved: medical_ct_diagnostic_model.keras")
    
    print("\n==========================================================")
    print("[+] ALL TRAINING SUCCESSFULLY COMPLETE!")
    print("    Download 'medical_ct_denoiser_full.keras' and")
    print("    'medical_ct_diagnostic_model.keras' to your local directory.")
    print("==========================================================")
