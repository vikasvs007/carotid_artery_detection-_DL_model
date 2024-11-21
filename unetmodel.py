import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2

# Define paths
image_dir = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\US images'  # Folder with artery images
mask_dir = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\Expert mask images'    # Folder with corresponding masks

# Parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 20
EPOCHS = 20
MODEL_PATH = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\model.h5'

# Load and preprocess the data
def load_images_and_masks(image_dir, mask_dir, img_height, img_width):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Check if the image exists
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Load and check the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image at: {img_path}")
            continue
        img = cv2.resize(img, (img_width, img_height)) / 255.0
        
        # Check if the mask exists
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue
        
        # Load and check the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask at: {mask_path}")
            continue
        mask = cv2.resize(mask, (img_width, img_height)) / 255.0
        
        images.append(img[..., np.newaxis])
        masks.append(mask[..., np.newaxis])
    
    return np.array(images), np.array(masks)

images, masks = load_images_and_masks(image_dir, mask_dir, IMG_HEIGHT, IMG_WIDTH)
print(f"Loaded {images.shape[0]} images and {masks.shape[0]} masks.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define UNet model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create U-Net model
model = unet_model()

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save the trained model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
