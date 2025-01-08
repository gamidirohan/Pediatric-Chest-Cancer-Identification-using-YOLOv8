from PIL import Image
import numpy as np

# Load the image
img_path = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\val_png\1a1274d7f2cec073698f33649e1d4fa9.png"
img = Image.open(img_path)

# Convert the image to a numpy array
img_array = np.array(img)

# Calculate the average pixel value
average_pixel_value = np.mean(img_array)

# Print the average pixel value
print("Average Pixel Value:", average_pixel_value)