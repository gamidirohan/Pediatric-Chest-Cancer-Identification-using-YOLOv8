import os
import pydicom
from PIL import Image

# Define the paths to your directories
dataset_path = r'C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0'
train_dcm_dir = os.path.join(dataset_path, 'train')
test_dcm_dir = os.path.join(dataset_path, 'test')
train_png_dir = os.path.join(dataset_path, 'train_png')
test_png_dir = os.path.join(dataset_path, 'test_png')

# Create the new PNG directories if they don't exist
os.makedirs(train_png_dir, exist_ok=True)
os.makedirs(test_png_dir, exist_ok=True)

# Convert DICOM files to PNG files and move them to the new directories
for dcm_dir, png_dir in [(train_dcm_dir, train_png_dir), (test_dcm_dir, test_png_dir)]:
    for filename in os.listdir(dcm_dir):
        if filename.endswith('.dicom'):
            dcm_path = os.path.join(dcm_dir, filename)
            png_path = os.path.join(png_dir, filename.replace('.dicom', '.png'))
            dcm_data = pydicom.dcmread(dcm_path)
            img = dcm_data.pixel_array
            final_image = Image.fromarray(img)
            final_image.save(png_path)
            print(f"Saved PNG file: {png_path}")

print("DICOM to PNG conversion complete.")