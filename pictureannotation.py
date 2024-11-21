import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image
original_image = cv2.imread(r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\US images\202201121748100022VAS_slice_1183.png')
original_image = cv2.resize(original_image, (128, 128))
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Convert grayscale to RGB for blending
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

# Load the overlay image (with or without alpha channel)
overlay_image = cv2.imread(r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\Expert mask images\202201121748100022VAS_slice_1393.png', cv2.IMREAD_UNCHANGED)
# Ensure the overlay image has the same size as the original image
overlay_image = cv2.resize(overlay_image, (original_image_rgb.shape[1], original_image_rgb.shape[0]))

# Check the number of channels in the overlay image
if len(overlay_image.shape) == 3 and overlay_image.shape[2] == 4:  # If overlay image has 4 channels (RGBA)
    alpha_channel = overlay_image[:, :, 3] / 255.0  # Extract the alpha channel and normalize it
    overlay_image_rgb = overlay_image[:, :, :3]  # Extract the RGB part of the overlay
else:
    alpha_channel = np.ones((overlay_image.shape[0], overlay_image.shape[1]))  # No alpha, full opacity
    overlay_image_rgb = overlay_image  # Just RGB (no alpha channel)

# Example: Annotate a specific region with a bounding box (replace with your annotation method)
# Let's assume the annotated area is a bounding box (x, y, width, height)
x, y, w, h = 10, 10, 100, 100  # Ensure the box fits within the resized 128x128 image

# Draw the bounding box on the original image (using a color like red for visibility)
annotated_image = original_image.copy()
cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Red box

# Now, blend the overlay image with the original image using the alpha channel
blended_image = original_image_rgb.astype(float)

# Loop over each RGB channel to blend based on alpha
for c in range(2):  # Loop over RGB channels
    blended_image[:, :, c] = (alpha_channel * overlay_image_rgb+ 
                              (1 - alpha_channel) * blended_image[:, :, c])

# Convert blended image back to uint8 for display
blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

# Display images using matplotlib
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Original Image with Annotation")
axs[0].axis('off')

axs[1].imshow(annotated_image, cmap='gray')
axs[1].set_title("Annotated Image (with Bounding Box)")
axs[1].axis('off')

axs[2].imshow(blended_image)
axs[2].set_title("Blended Image with Overlay")
axs[2].axis('off')

plt.show()
