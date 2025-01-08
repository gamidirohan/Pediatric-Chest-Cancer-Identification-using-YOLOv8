# Pediatric Chest Cancer Identification using YOLOv8

This repository contains the code and resources for identifying thoracic diseases in pediatric chest X-rays using the YOLOv8 model. The dataset used is **VinDr-PCXR: An open, large-scale pediatric chest X-ray dataset for interpretation of common thoracic diseases**.

---

## Files and Their Purpose

### Preprocessing and Dataset Preparation
1. **convert_dicome_to_png.py**  
   Converts DICOM format files into PNG images for easier processing.

2. **split_test_and_val.py**  
   Splits the dataset into test and validation sets.

3. **YOLO_format.py**  
   Converts `.csv` label files into `.txt` format for compatibility with YOLO.

4. **yolo_dataset_structure.py**  
   Structures the dataset according to the YOLO requirements. (Double-check if this is redundant or if it's the same functionality as another script.)

5. **dataset.yaml**  
   Configuration file for YOLO, defining class names and dataset paths.

### Model Training and Testing
1. **YOLOV8_train.py**  
   Script to train the YOLOv8 model using the prepared dataset.

2. **visualize_predictions.py**  
   Visualizes the predictions made by the YOLOv8 model on test data.

3. **YOLOV8_test_model.py**  
   Script for testing the YOLOv8 model to verify its functionality.

### Annotations
- **annotations_test.txt**  
- **annotations_train.txt**  
   Processed `.txt` files containing label annotations for the test and training datasets.

### Normalization (Optional)
1. **Normalization.py**  
   Normalizes images to standardize pixel intensity values (optional).

2. **average_pixel_value.py**  
   Computes the average pixel value across images for normalization reference.

### Folder Structure
- **folder_structure.py**  
   A reference script showing the required folder structure for organizing the dataset and model outputs.

---

## Folder Structure Reference
Ensure the following folder structure is used:

```plaintext
project/
│
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│
├── annotations/
│   ├── annotations_train.txt
│   ├── annotations_test.txt
│
├── models/
│   ├── YOLOv8/
│
├── scripts/
│   ├── convert_dicome_to_png.py
│   ├── split_test_and_val.py
│   ├── YOLO_format.py
│   ├── yolo_dataset_structure.py
│   ├── dataset.yaml
│   ├── YOLOV8_train.py
│   ├── YOLOV8_test_model.py
│   ├── visualize_predictions.py
│   ├── folder_structure.py
│   ├── Normalization.py
│   ├── average_pixel_value.py
│
