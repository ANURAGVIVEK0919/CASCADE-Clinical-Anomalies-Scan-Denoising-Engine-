import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DATA_DIR = r"d:\DATA"
CSV_PATH = os.path.join(BASE_DATA_DIR, "metadata.csv")
RES_FOLDER = "Preprocessed_256x256/256" 

IMG_SIZE = (256, 256)
CHANNELS = 1 
BATCH_SIZE = 8
EPOCHS = 30 # Poore data ke liye 30-50 epochs kaafi hain

# ──────────────────────────────────────────────
# DATA GENERATOR (PROFESSIONAL WAY)
# ──────────────────────────────────────────────
def parse_image(noisy_path, clean_path):
    # Noisy load karo
    noisy = tf.io.read_file(noisy_path)
    noisy = tf.image.decode_png(noisy, channels=1)
    noisy = tf.image.resize(noisy, IMG_SIZE)
    noisy = tf.cast(noisy, tf.float32) / 255.0
    
    # Clean load karo
    clean = tf.io.read_file(clean_path)
    clean = tf.image.decode_png(clean, channels=1)
    clean = tf.image.resize(clean, IMG_SIZE)
    clean = tf.cast(clean, tf.float32) / 255.0
    
    return noisy, clean

def get_dataset(csv_path, base_dir, res_folder):
    noisy_paths = []
    clean_paths = []
    
    print(f"[*] Reading all paths from {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_noisy = row['Quarter Dose Filepath'].replace("512/", res_folder + "/")
            rel_clean = row['Full Dose Filepath'].replace("512/", res_folder + "/")
            
            full_noisy = os.path.join(base_dir, rel_noisy).replace("/", "\\")
            full_clean = os.path.join(base_dir, rel_clean).replace("/", "\\")
            
            if os.path.exists(full_noisy) and os.path.exists(full_clean):
                noisy_paths.append(full_noisy)
                clean_paths.append(full_clean)
                
    print(f"[*] Total valid pairs found: {len(noisy_paths)}")
    
    dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, len(noisy_paths)

# ──────────────────────────────────────────────
# U-NET ARCHITECTURE
# ──────────────────────────────────────────────
def build_unet(input_shape):
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        return x

    inputs = layers.Input(input_shape)
    f1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D(2)(f1)
    f2 = conv_block(p1, 64); p2 = layers.MaxPooling2D(2)(f2)
    b = conv_block(p2, 128)
    u2 = layers.UpSampling2D(2)(b); u2 = layers.concatenate([u2, f2]); c2 = conv_block(u2, 64)
    u1 = layers.UpSampling2D(2)(c2); u1 = layers.concatenate([u1, f1]); c1 = conv_block(u1, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c1)

    model = models.Model(inputs, outputs, name="Medical_U-Net")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: metadata.csv nahi mili: {CSV_PATH}")
    else:
        dataset, total_count = get_dataset(CSV_PATH, BASE_DATA_DIR, RES_FOLDER)
        
        # Train/Val split (Take first 10% for validation)
        val_size = int(0.1 * (total_count // BATCH_SIZE))
        val_ds = dataset.take(val_size)
        train_ds = dataset.skip(val_size)
        
        model = build_unet((IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
        model.summary()
        
        print(f"\n[*] Poore dataset par training shuru ho rahi hai...")
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        model.save("medical_ct_denoiser_full.keras")
        print("[*] Model saved: medical_ct_denoiser_full.keras")
        
        # Show one test sample
        for n, c in val_ds.take(1):
            pred = model.predict(n)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(n[0].numpy().squeeze(), cmap='gray'); plt.title("Noisy (QD)")
            plt.subplot(1, 3, 2); plt.imshow(pred[0].squeeze(), cmap='gray'); plt.title("AI Cleaned")
            plt.subplot(1, 3, 3); plt.imshow(c[0].numpy().squeeze(), cmap='gray'); plt.title("Original (FD)")
            plt.savefig("full_data_result_sample.png")
            print("[*] Sample saved: full_data_result_sample.png")
            plt.show()
            break
