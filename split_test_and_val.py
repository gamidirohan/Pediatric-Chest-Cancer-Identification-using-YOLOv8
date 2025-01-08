import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your dataset directories
test_dir = r'C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\test_png'
val_dir = r'C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\val_png'
new_test_dir = r'C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\new_test_png'

# Create directories if they don't exist
os.makedirs(val_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

# Get a list of all files in the test directory
all_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

# Split the files into 20% validation and 80% test
val_files, new_test_files = train_test_split(all_files, test_size=0.8, random_state=42)

# Move files to the validation directory
for file in val_files:
    shutil.move(os.path.join(test_dir, file), os.path.join(val_dir, file))

# Move files to the new test directory
for file in new_test_files:
    shutil.move(os.path.join(test_dir, file), os.path.join(new_test_dir, file))

print(f"Validation set created with {len(val_files)} images.")
print(f"New test set created with {len(new_test_files)} images.")
