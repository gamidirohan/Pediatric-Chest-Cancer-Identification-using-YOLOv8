import os
import pandas as pd

# Paths to input files and output folders
annotations_train_path = "C:/dataset/vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0/annotations_train.csv"
annotations_val_path = "C:/dataset/vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0/annotations_val.csv"
output_labels_train = "C:/dataset/vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0/data/labels/train"
output_labels_val = "C:/dataset/vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0/data/labels/val"

# Ensure output directories exist
os.makedirs(output_labels_train, exist_ok=True)
os.makedirs(output_labels_val, exist_ok=True)

def convert_to_yolo_format(row, image_width, image_height):
    """Convert bounding box data to YOLO format."""
    x_center = (row['x_min'] + row['x_max']) / 2 / image_width
    y_center = (row['y_min'] + row['y_max']) / 2 / image_height
    width = (row['x_max'] - row['x_min']) / image_width
    height = (row['y_max'] - row['y_min']) / image_height
    
    # Validate that coordinates are within the range [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        return None  # Return None for invalid bounding boxes
    return f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_annotations(input_csv, output_folder, image_width=1024, image_height=1024):
    """Process annotations and create YOLO format label files."""
    # Load the CSV file
    annotations = pd.read_csv(input_csv)

    # Group by image_id
    grouped = annotations.groupby('image_id')

    for image_id, group in grouped:
        # Generate YOLO format lines for all bounding boxes in this image
        yolo_lines = []
        for _, row in group.iterrows():
            yolo_line = convert_to_yolo_format(row, image_width, image_height)
            if yolo_line:
                yolo_lines.append(yolo_line)
            else:
                print(f"WARNING: Skipping invalid bounding box for image {image_id}")

        # Write to corresponding label file if there are valid annotations
        if yolo_lines:
            label_file_path = os.path.join(output_folder, f"{image_id}.txt")
            with open(label_file_path, 'w') as f:
                f.write("\n".join(yolo_lines))
        else:
            print(f"WARNING: No valid annotations for image {image_id}, skipping label file.")

# Process train and validation annotations
process_annotations(annotations_train_path, output_labels_train)
process_annotations(annotations_val_path, output_labels_val)

print("Conversion to YOLO format completed.")
