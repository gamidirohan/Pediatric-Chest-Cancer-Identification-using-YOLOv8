import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the saved model
model = YOLO('yolov8n.pt')

# Load a test image
test_img_path = 'path_to_test_image.jpg'
test_img = cv2.imread(test_img_path)

# Get predictions
predictions = model.predict(test_img)

# Loop through predictions and draw bounding boxes
for prediction in predictions:
    x, y, w, h, confidence, class_id = prediction
    class_name = model.names[int(class_id)]
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(test_img, f"{class_name}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the output
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.show()