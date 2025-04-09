import cv2
import numpy as np

# Load the mask image
mask = cv2.imread(r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\results\new 100\1186.png", 0)  # Load in grayscale

# Define the kernel size (adjust as needed)
kernel = np.ones((5, 5), np.uint8)

# Apply erosion
eroded_mask = cv2.erode(mask, kernel, iterations=1)

# Save or visualize the eroded mask
cv2.imwrite('narrowed_mask1.png', eroded_mask)
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create a blank mask
# narrowed_mask = np.zeros_like(mask)

# # Iterate over contours and draw them with scaling
# for contour in contours:
#     # Optionally scale down the contour here if needed
#     # Draw the new, smaller contour
#     cv2.drawContours(narrowed_mask, [contour], -1, (255), thickness=-1)

# # Save or display the result
# cv2.imwrite('narrowed_mask2.png', narrowed_mask)
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

# Threshold to create a narrower mask
_, narrowed_mask = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Convert back to uint8 type
narrowed_mask = np.uint8(narrowed_mask)

# Save or display the result
cv2.imwrite('narrowed_mask3.png', narrowed_mask)