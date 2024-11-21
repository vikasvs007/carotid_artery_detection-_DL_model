


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\annotedimage\WhatsApp Image 2024-11-10 at 3.11.27 AM (1).jpeg'  # Replace with your actual path
# image = cv2.imread(image_path)

# # Convert the image to HSV color space for color filtering
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define HSV ranges for yellow and blue
# yellow_lower = np.array([20, 100, 100])
# yellow_upper = np.array([30, 255, 255])
# blue_lower = np.array([90, 100, 100])
# blue_upper = np.array([130, 255, 255])

# # Create masks for yellow and blue contours
# yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
# blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

# # Find contours for yellow and blue regions
# yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Select the largest contour for yellow and blue (assuming they represent the boundaries of interest)
# yellow_contour = max(yellow_contours, key=cv2.contourArea) if yellow_contours else None
# blue_contour = max(blue_contours, key=cv2.contourArea) if blue_contours else None

# # Retrieve and print yellow boundary pixel coordinates if found
# if yellow_contour is not None:
#     yellow_boundary_pixels = [tuple(point[0]) for point in yellow_contour]  # Convert to list of (x, y) tuples
#     print("Yellow Boundary Pixel Coordinates:")
#     for pixel in yellow_boundary_pixels:
#         print(pixel)
# else:
#     print("Yellow contour was not found.")

# # Retrieve and print blue boundary pixel coordinates if found
# if blue_contour is not None:
#     blue_boundary_pixels = [tuple(point[0]) for point in blue_contour]  # Convert to list of (x, y) tuples
#     print("\nBlue Boundary Pixel Coordinates:")
#     for pixel in blue_boundary_pixels:
#         print(pixel)
# else:
#     print("Blue contour was not found.")

# # Calculate the distance between the yellow and blue contours if both are found
# if yellow_contour is not None and blue_contour is not None:
#     # Initialize a list to store distances
#     distances = []
    
#     # For each point in the yellow contour, find the closest point on the blue contour
#     for point in yellow_contour:
#         yellow_point = tuple(point[0])  # Flatten the point to (x, y)
        
#         # Find the minimum Euclidean distance from the yellow point to any point on the blue contour
#         min_distance = float('inf')
#         for blue_point in blue_contour:
#             blue_point = tuple(blue_point[0])  # Flatten the point to (x, y)
#             dist = np.linalg.norm(np.array(yellow_point) - np.array(blue_point))  # Calculate Euclidean distance
#             if dist < min_distance:
#                 min_distance = dist
        
#         # Append the minimum distance to the list
#         distances.append(min_distance)
    
#     # Calculate average gap distance between the contours
#     avg_distance = np.mean(distances)
#     print(f"\nAverage gap distance between yellow and blue boundaries: {avg_distance:.2f} pixels")
# else:
#     print("One or both of the contours were not found.")

# # Visualize the contours and gap
# contour_image = image.copy()
# if yellow_contour is not None:
#     cv2.drawContours(contour_image, [yellow_contour], -1, (0, 255, 255), 2)  # Draw yellow contour in yellow color
# if blue_contour is not None:
#     cv2.drawContours(contour_image, [blue_contour], -1, (255, 0, 0), 2)  # Draw blue contour in blue color

# # Display the image with contours
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
# plt.title("Contours of Inner Wall (Yellow) and Residual Lumen (Blue)")
# plt.axis('off')
# plt.show()





import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directory containing your 15 images
dataset_dir = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\annotedimage'  # Replace with your actual directory path

# Define HSV ranges for yellow and blue
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
blue_lower = np.array([90, 100, 100])
blue_upper = np.array([130, 255, 255])

# List to store average distances and categories for each image
average_distances = []
categories = []

# Process each image in the dataset directory
for img_file in sorted(os.listdir(dataset_dir)):
    # Load the image
    image_path = os.path.join(dataset_dir, img_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Convert the image to HSV color space for color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for yellow and blue contours
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Find contours for yellow and blue regions
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour for yellow and blue (assuming they represent the boundaries of interest)
    yellow_contour = max(yellow_contours, key=cv2.contourArea) if yellow_contours else None
    blue_contour = max(blue_contours, key=cv2.contourArea) if blue_contours else None

    # Calculate the distance between the yellow and blue contours if both are found
    if yellow_contour is not None and blue_contour is not None:
        distances = []
        
        # For each point in the yellow contour, find the closest point on the blue contour
        for point in yellow_contour:
            yellow_point = tuple(point[0])  # Flatten the point to (x, y)
            
            # Find the minimum Euclidean distance from the yellow point to any point on the blue contour
            min_distance = float('inf')
            for blue_point in blue_contour:
                blue_point = tuple(blue_point[0])  # Flatten the point to (x, y)
                dist = np.linalg.norm(np.array(yellow_point) - np.array(blue_point))  # Calculate Euclidean distance
                if dist < min_distance:
                    min_distance = dist
            
            # Append the minimum distance to the list
            distances.append(min_distance)
        
        # Calculate average gap distance between the contours for this image
        avg_distance = np.mean(distances)
        average_distances.append(avg_distance)

        # Categorize based on average distance
        if avg_distance > 60:
            category = "Healthy"
        elif avg_distance <= 20:
            category = "Danger"
        else:
            category = "Moderate"

        categories.append(category)
        print(f"{img_file}: Average gap distance = {avg_distance:.2f} pixels - {category}")
    else:
        print(f"Contours not found for image: {img_file}")
        average_distances.append(np.nan)  # Use NaN for missing values
        categories.append("No Data")

# Plot the average gap distances for all images with category-based color coding
plt.figure(figsize=(12, 6))

# Assign colors based on the categories
colors = ['g' if cat == 'Healthy' else 'r' if cat == 'Danger' else 'orange' for cat in categories]
plt.bar(range(1, len(average_distances) + 1), average_distances, color=colors)

# Adding labels
plt.xlabel("Image Index")
plt.ylabel("Average Gap Distance (pixels)")
plt.title("Average Gap Distance between Yellow and Blue Boundaries")
plt.xticks(range(1, len(average_distances) + 1))  # Set x-ticks to image indices
plt.grid(True)

# Adding legend
plt.plot([], [], color="g", label="Healthy (> 60 pixels)")
plt.plot([], [], color="orange", label="Moderate (40-60 pixels)")
plt.plot([], [], color="r", label="Danger (≤ 20 pixels)")
plt.legend()

# Display the plot
plt.show()
