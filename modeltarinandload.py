from PIL import Image
import numpy as np

# Load the actual image
X_image = np.array(Image.open(r'C:\Users\vikas\Downloads\Common Carotid Artery Ultrasound Images\Common Carotid Artery Ultrasound Images\US images\202202071355560051VAS_slice_673.png').convert('L').resize((128, 128)))
X_image = np.expand_dims(X_image, axis=(0, -1)) / 255.0  # Normalize and add batch/channel dimensions

# Load the actual mask
Y_mask = np.array(Image.open(r'C:\Users\vikas\Downloads\Common Carotid Artery Ultrasound Images\Common Carotid Artery Ultrasound Images\US images\202202071355560051VAS_slice_673.png').convert('L').resize((128, 128)))
Y_mask = (Y_mask > 127).astype(np.float32)  # Convert to binary (1 for mask, 0 for background)
Y_mask = np.expand_dims(Y_mask, axis=(0, -1))  # Add batch/channel dimensions


import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(X_image[0, :, :, 0], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(Y_mask[0, :, :, 0], cmap="gray")

plt.show()
