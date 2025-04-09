import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_image_and_mask(image_path, mask_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return
    
    # Display the image and mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    # Print unique values in the mask
    unique_values = np.unique(mask)
    print(f"Unique values in the mask: {unique_values}")

if __name__ == "__main__":
    images_directory = r"D:\Thesis\datasets\main_DATASET\AUGMENTED\img"
    masks_directory = (r"D:\Thesis\datasets\main_DATASET\AUGMENTED\msks")
    
    
    # List all image files
    image_files = [f for f in os.listdir(images_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Verify each image and its corresponding mask
    for image_file in image_files:
        # Construct the corresponding mask filename
        mask_file = image_file.replace('image', 'mask')
        image_path = os.path.join(images_directory, image_file)
        mask_path = os.path.join(masks_directory, mask_file)
        
        if os.path.exists(mask_path):
            print(f"Verifying {image_file} and {mask_file}")
            verify_image_and_mask(image_path, mask_path)
        else:
            print(f"No mask found for image: {image_file}")
