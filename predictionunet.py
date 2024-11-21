import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define the paths
MODEL_PATH = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\model.h5'
image_path = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\DATASET1\frame-11.jpg'

# Load the trained model
model = load_model(MODEL_PATH)

# Preprocess the new image
IMG_HEIGHT, IMG_WIDTH = 128, 128

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img = img[..., np.newaxis]  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load and preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make the prediction
predicted_mask = model.predict(preprocessed_image)[0]  # Remove batch dimension

# Apply threshold to create binary mask
threshold = 0.3
binary_mask = (predicted_mask > threshold).astype(np.uint8)

# Resize mask and original image for visualization
binary_mask_resized = cv2.resize(binary_mask[:, :, 0], (IMG_WIDTH, IMG_HEIGHT))
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_image_resized = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in descending order to get the three largest
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Create a color overlay for annotations
overlay = cv2.cvtColor(original_image_resized, cv2.COLOR_GRAY2BGR)

# Draw the contours in reverse order (smallest first) so the red boundary is drawn on top
if len(sorted_contours) > 0:
    cv2.drawContours(overlay, [sorted_contours[2]], -1, (255, 0, 0), 1)  # Affected area in blue
if len(sorted_contours) > 1:
    cv2.drawContours(overlay, [sorted_contours[1]], -1, (0, 255, 255), 1)  # Inner boundary in yellow
if len(sorted_contours) > 2:
    cv2.drawContours(overlay, [sorted_contours[0]], -1, (0, 0, 255), 1)  # Outer boundary in red (on top)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image_resized, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Annotated Boundaries")
plt.axis('off')

plt.show()
