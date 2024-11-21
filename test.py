import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths
us_folder = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\US images'
mask_folder = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\Expert mask images'
output_folder = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\blendedimage'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to superimpose mask onto ultrasound image
def superimpose_images(us_image, mask_image):
    # Resize images to match
    us_image = cv2.resize(us_image, (128, 128))
    mask_image = cv2.resize(mask_image, (128, 128))

    # Convert ultrasound image to grayscale if needed and then to RGB
    if len(us_image.shape) == 2:
        us_image_rgb = cv2.cvtColor(us_image, cv2.COLOR_GRAY2RGB)
    else:
        us_image_rgb = cv2.cvtColor(cv2.cvtColor(us_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

    # Handle mask image with or without alpha channel
    if len(mask_image.shape) == 3 and mask_image.shape[2] == 4:  # RGBA image
        alpha_channel = mask_image[:, :, 3] / 255.0
        mask_image_rgb = mask_image[:, :, :3]
    else:  # RGB image without alpha
        alpha_channel = np.ones((mask_image.shape[0], mask_image.shape[1]), dtype=np.float32)
        mask_image_rgb = mask_image

    # Blend the images
    blended_image = us_image_rgb.astype(float)
    for c in range(3):  # Loop over RGB channels
        blended_image[:, :, c] = (alpha_channel * mask_image_rgb +
                                  (1 - alpha_channel) * blended_image[:, :, c])

    # Convert blended image back to uint8
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

# Process each ultrasound and corresponding mask image
for us_filename in os.listdir(us_folder):
    # Construct full paths for the ultrasound and mask images
    us_path = os.path.join(us_folder, us_filename)
    mask_path = os.path.join(mask_folder, us_filename)

    # Check if both ultrasound and mask images exist
    if os.path.exists(us_path) and os.path.exists(mask_path):
        # Load images
        us_image = cv2.imread(us_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Superimpose mask on ultrasound image
        blended_image = superimpose_images(us_image, mask_image)

        # Save the blended image
        output_path = os.path.join(output_folder, f"blended_{us_filename}.png")
        cv2.imwrite(output_path, blended_image)  # Save as an image file
        print(f"Saved blended image: {output_path}")
    else:
        print(f"Skipping {us_filename}: Mask or US image not found.")
