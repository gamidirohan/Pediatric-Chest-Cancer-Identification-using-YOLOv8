import os
import numpy as np
from PIL import Image
from skimage import exposure
import cv2

# Define the input and output folders
input_train_folder = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\train_png\images"
input_val_folder = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\val_png\images"

output_train_folder = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\train_png_normalized"
output_val_folder = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\val_png_normalized"

# Create the output folders if they don't exist
if not os.path.exists(output_train_folder):
    os.makedirs(output_train_folder)

if not os.path.exists(output_val_folder):
    os.makedirs(output_val_folder)

# Function to apply CLAHE normalization
def apply_clahe(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    avg_pixel_value = np.mean(img)

    if avg_pixel_value < 500:
        cv2.imwrite(output_path, img)
    else:
        clahe_img = exposure.equalize_adapthist(img, clip_limit=0.03)
        clahe_img = (clahe_img * 255).astype(np.uint8)
        cv2.imwrite(output_path, clahe_img)

# Apply CLAHE normalization to images in the input folders
for folder, output_folder in [(input_train_folder, output_train_folder), (input_val_folder, output_val_folder)]:
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, filename)
            apply_clahe(image_path, output_path)
            print(f"Normalized {filename} and saved to {output_folder}")