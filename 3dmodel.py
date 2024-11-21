
import SimpleITK as sitk
import numpy as np
import os
from mayavi import mlab

# Load the first image to determine target dimensions
folder_path = r"C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\DATASET1"  # Replace with your folder path
image_files = sorted(os.listdir(folder_path))
first_image_path = os.path.join(folder_path, image_files[0])
first_image = sitk.ReadImage(first_image_path)

# Convert the first image to grayscale if it's not already
first_image = sitk.Cast(sitk.VectorIndexSelectionCast(first_image, 0), sitk.sitkFloat32)
target_size = sitk.GetArrayFromImage(first_image).shape

# Initialize a list to store resized images
image_slices = []

for filename in image_files:
    slice_path = os.path.join(folder_path, filename)
    slice_image = sitk.ReadImage(slice_path)

    # Convert the slice to grayscale
    slice_image = sitk.Cast(sitk.VectorIndexSelectionCast(slice_image, 0), sitk.sitkFloat32)

    # Resize each image to the target size
    resized_image = sitk.Resample(slice_image, first_image)  # Resampling to the same size as the first image
    slice_array = sitk.GetArrayFromImage(resized_image)
    image_slices.append(slice_array)

# Stack resized images to create a 3D volume
volume = np.stack(image_slices, axis=-1)

# Print the shape of the volume to ensure it's 3D
print("volume.shape:", volume.shape)

# Proceed with your visualization code if the shape is correct
if len(volume.shape) == 3:
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.contour3d(volume, contours=10, opacity=0.5, colormap='gray')
    mlab.show()
else:
    print("Error: The volume is not 3D.")