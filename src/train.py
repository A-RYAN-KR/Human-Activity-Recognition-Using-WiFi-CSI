import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm

# Import local modules
from src.data_loader import load_dataset, augment_data
from src.model import build_model

# --- Configuration ---
SEQ_LEN = 384
NUM_CLASSES = 16
BATCH_SIZE = 32
EPOCHS = 120
DATA_ROOT = "WiAR" # Assumes user cloned WiAR dataset into this folder

def main():
    # 1. Load Data
    print(f"Looking for data in: {DATA_ROOT}")
    X_raw_list, y_list = load_dataset(DATA_ROOT, SEQ_LEN)
    
    if len(X_raw_list) == 0:
        print("No data found. Please run: git clone https://github.com/linteresa/WiAR.git")
        return

    # 2. Alignment (Padding features to max dimension)
    max_features = 0
    for seq in X_raw_list:
        if seq.shape[1] > max_features:
            max_features = seq.shape[1]
    
    print(f"Aligning features to max dimension: {max_features}")
    X_aligned = []
    for seq in X_raw_list:
        T, F = seq.shape
        if F < max_features:
            pad_width = ((0, 0), (0, max_features - F))
            seq_padded = np.pad(seq, pad_width, mode='constant')
            X_aligned.append(seq_padded)
        else:
            X_aligned.append(seq[:, :max_features])

    X = np.array(X_aligned, dtype='float32')
    y = np.array(y_list, dtype='int32')
    
    # 3. Split
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Normalize
    print("Standardizing data...")
    mean_train = X_train_raw.mean(axis=(1, 2), keepdims=True)
    std_train = X_train_raw.std(axis=(1, 2), keepdims=True) + 1e-6
    X_train_raw = (X_train_raw - mean_train) / std_train

    mean_test = X_test.mean(axis=(1, 2), keepdims=True)
    std_test = X_test.std(axis=(1, 2), keepdims=True) + 1e-6
    X_test = (X_test - mean_test) / std_test

    # 5. Augmentation
    print("Augmenting Training Data...")
    X_train_aug = []
    y_train_aug = []
    for i in tqdm(range(len(X_train_raw)), desc="Augmenting"):
        X_train_aug.append(X_train_raw[i])
        y_train_aug.append(y_train_raw[i])
        
        aug = augment_data(X_train_raw[i])
        X_train_aug.append(aug)
        y_train_aug.append(y_train_raw[i])

    X_train = np.array(X_train_aug, dtype='float32')
    y_train = np.array(y_train_aug, dtype='float32')

    # 6. Build and Train
    model = build_model(input_shape=(SEQ_LEN, max_features), num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ModelCheckpoint('models/best_wiar_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        shuffle=True
    )

    print("Training Complete. Model saved to models/best_wiar_model.h5")

if __name__ == "__main__":
    main()
