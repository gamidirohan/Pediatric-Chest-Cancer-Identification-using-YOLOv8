import pandas as pd
import os

# Function to convert annotations to YOLO format
def convert_annotations(annotations_file, output_dir, class_mapping, image_width, image_height):
    # Load the annotations data
    annotations = pd.read_csv(annotations_file, sep=' ', header=None)
    annotations.columns = ['image_id', 'radiologist_id', 'label_name', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence']

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert the annotations
    for _, row in annotations.iterrows():
        image_id = row['image_id']
        class_id = class_mapping.get(row['label_name'], -1)
        if class_id == -1:
            continue  # Skip if the class is not mapped

        # Convert to YOLO format
        x_min = row['x_min']
        y_min = row['y_min']
        x_max = row['x_max']
        y_max = row['y_max']

        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Create the output label file
        label_file_path = os.path.join(output_dir, f"{image_id}.txt")
        with open(label_file_path, 'a') as label_file:
            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Class mapping (replace with actual mappings)
class_mapping = {
    "Boot-shaped heart": 0,
    "Peribronchovascular interstitial opacity": 1,
    "Reticulonodular opacity": 2,
    "Bronchial thickening": 3,
    "Enlarged PA": 4,
    "Cardiomegaly": 5,
    "Other opacity": 6,
    "Other lesion": 7,
    "Diffuse aveolar opacity": 8,
    "Consolidation": 9,
    "Mediastinal shift": 10,
    "Anterior mediastinal mass": 12
}

# Image dimensions (adjust or read dynamically)
image_width = 1024  # Example width
image_height = 1024  # Example height

# Convert training annotations
train_annotations_file = 'annotations_train.txt'
train_output_dir = 'labels/train'
convert_annotations(train_annotations_file, train_output_dir, class_mapping, image_width, image_height)
print(f"Training labels created in '{train_output_dir}'")

# Convert test annotations
test_annotations_file = 'annotations_test.txt'
test_output_dir = 'labels/test'
convert_annotations(test_annotations_file, test_output_dir, class_mapping, image_width, image_height)
print(f"Test labels created in '{test_output_dir}'")
