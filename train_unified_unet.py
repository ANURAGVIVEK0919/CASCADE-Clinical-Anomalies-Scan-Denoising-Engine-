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
MODEL_OUT_PATH = r"d:\Medical-Denoise-AI-CT-Scan-Restoration--main\Medical-Denoise-AI-CT-Scan-Restoration--main\medical_ct_denoiser_full.keras"

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 30  # High-fidelity deep training epochs

# ────────────────────────────────────────────────────────
# HIGH-PERFORMANCE DATA PIPELINE
# ────────────────────────────────────────────────────────
def parse_image_pairs_direct(noisy_abs, clean_abs):
    # Read noisy slice
    noisy_img = tf.io.read_file(noisy_abs)
    noisy_img = tf.image.decode_png(noisy_img, channels=1)
    noisy_img = tf.image.resize(noisy_img, IMG_SIZE)
    noisy_img = tf.cast(noisy_img, tf.float32) / 255.0
    
    # Read clean slice
    clean_img = tf.io.read_file(clean_abs)
    clean_img = tf.image.decode_png(clean_img, channels=1)
    clean_img = tf.image.resize(clean_img, IMG_SIZE)
    clean_img = tf.cast(clean_img, tf.float32) / 255.0
    
    return noisy_img, clean_img

def get_denoising_datasets():
    # Scan all directories in separated_scans to pair noisy and clear images
    separated_dir = os.path.join(DATA_DIR, "separated_scans")
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
                
                # Clean counterpart
                clean_f = f.replace('_noisy', '_clean')
                clean_path = os.path.join(clear_dir, clean_f)
                
                if not os.path.exists(clean_path):
                    clean_f = f.replace('noisy', 'clear')
                    clean_path = os.path.join(clear_dir, clean_f)
                    
                if os.path.exists(clean_path):
                    noisy_paths.append(noisy_path)
                    clean_paths.append(clean_path)
                    
    print(f"[*] Loaded {len(noisy_paths)} paired images from separated_scans (including augmented files).")
    
    # Shuffle and split into train/val (90% train, 10% val)
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
        
    train_ds = make_tf_dataset(train_noisy, train_clean).shuffle(300).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(val_noisy, val_clean).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# ────────────────────────────────────────────────────────
# SYMMETRIC U-NET AUTOENCODER ARCHITECTURE
# ────────────────────────────────────────────────────────
def build_unet(input_shape=(256, 256, 1)):
    """
    Builds a lightweight 2D Convolutional U-Net Autoencoder:
    - Downsampling/Encoder: extracts hierarchical noise-invariant features.
    - Bottleneck: latent representations of clean anatomical details.
    - Upsampling/Decoder + Skip Connections: maps spatial details back to clean pixels.
    """
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
        
    inputs = layers.Input(input_shape)
    
    # Encoder Block 1
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Encoder Block 2
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = conv_block(p2, 128)
    
    # Decoder Block 1
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = conv_block(u4, 64)
    
    # Decoder Block 2
    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = conv_block(u5, 32)
    
    # Output projection layer (Sigmoid maps pixels between 0.0 and 1.0)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet_CT_Denoiser")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


# ────────────────────────────────────────────────────────
# EXECUTION PIPELINE
# ────────────────────────────────────────────────────────
def run_training():
    separated_dir = os.path.join(DATA_DIR, "separated_scans")
    if not os.path.exists(separated_dir):
        print(f"[!] Error: separated_scans directory not found at {separated_dir}. Run separation script first.")
        return
        
    # Load dataset pipelines
    train_ds, val_ds = get_denoising_datasets()
    
    # Build Model
    print("[*] Compiling symmetric U-Net Denoiser...")
    model = build_unet()
    model.summary()
    
    # Train with callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    print(f"[*] Starting U-Net training for {EPOCHS} epochs with checkpoints and decay...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save
    print(f"[*] Saving trained weights to: {MODEL_OUT_PATH}")
    model.save(MODEL_OUT_PATH)
    
    # Save training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('U-Net Denoising Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, "unet_training_loss.png"))
    print("[*] Training curves saved successfully: unet_training_loss.png")

if __name__ == "__main__":
    run_training()
