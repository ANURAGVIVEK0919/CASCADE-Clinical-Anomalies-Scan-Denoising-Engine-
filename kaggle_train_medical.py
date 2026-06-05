import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


BASE_DATA_DIR = "/kaggle/input/ct-low-dose-reconstruction" 
CSV_PATH = os.path.join(BASE_DATA_DIR, "metadata.csv")

IMG_SIZE = (256, 256)
CHANNELS = 1 
BATCH_SIZE = 16  # Kaggle GPU can handle larger batch size
EPOCHS = 30

# ──────────────────────────────────────────────
# KAGGLE DATA PIPELINE
# ──────────────────────────────────────────────
def parse_image(noisy_path, clean_path):
    noisy = tf.io.read_file(noisy_path)
    noisy = tf.image.decode_png(noisy, channels=1)
    noisy = tf.image.resize(noisy, IMG_SIZE)
    noisy = tf.cast(noisy, tf.float32) / 255.0
    
    clean = tf.io.read_file(clean_path)
    clean = tf.image.decode_png(clean, channels=1)
    clean = tf.image.resize(clean, IMG_SIZE)
    clean = tf.cast(clean, tf.float32) / 255.0
    
    return noisy, clean

def get_kaggle_dataset(csv_path, base_dir):
    noisy_paths = []
    clean_paths = []
    
    print(f"[*] Reading paths from {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Kaggle path mapping (direct from CSV)
            full_noisy = os.path.join(base_dir, row['Quarter Dose Filepath'])
            full_clean = os.path.join(base_dir, row['Full Dose Filepath'])
            
            if os.path.exists(full_noisy):
                noisy_paths.append(full_noisy)
                clean_paths.append(full_clean)
                
    print(f"[*] Found {len(noisy_paths)} valid pairs on Kaggle.")
    
    dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))
    dataset = dataset.shuffle(2000)
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, len(noisy_paths)

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────
def build_unet(input_shape):
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        return x

    inputs = layers.Input(input_shape)
    # Encoder
    f1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D(2)(f1)
    f2 = conv_block(p1, 64); p2 = layers.MaxPooling2D(2)(f2)
    # Bridge
    b = conv_block(p2, 128)
    # Decoder
    u2 = layers.UpSampling2D(2)(b); u2 = layers.concatenate([u2, f2]); c2 = conv_block(u2, 64)
    u1 = layers.UpSampling2D(2)(c2); u1 = layers.concatenate([u1, f1]); c1 = conv_block(u1, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c1)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ──────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("ERROR: Dataset link check karein, metadata.csv nahi mili.")
    else:
        dataset, total_count = get_kaggle_dataset(CSV_PATH, BASE_DATA_DIR)
        
        # Train/Val split
        val_size = int(0.1 * (total_count // BATCH_SIZE))
        val_ds = dataset.take(val_size)
        train_ds = dataset.skip(val_size)
        
        model = build_unet((IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
        
        print("\n[*] Kaggle GPU Training Starting...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        # Save model in Kaggle Output
        model.save("medical_ct_denoiser_kaggle.keras")
        print("[*] Training Complete! Model saved as medical_ct_denoiser_kaggle.keras")
        
        # Plot result
        for n, c in val_ds.take(1):
            pred = model.predict(n)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(n[0].numpy().squeeze(), cmap='gray'); plt.title("Noisy")
            plt.subplot(1, 3, 2); plt.imshow(pred[0].squeeze(), cmap='gray'); plt.title("Denoised")
            plt.subplot(1, 3, 3); plt.imshow(c[0].numpy().squeeze(), cmap='gray'); plt.title("Ground Truth")
            plt.show()
