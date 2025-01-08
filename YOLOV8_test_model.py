from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.yaml")

# Load an example image
img = "https://ultralytics.com/images/zidane.jpg"

# Perform inference
results = model(img)

# Print the results
print(results)