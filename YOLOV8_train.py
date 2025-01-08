#######
# 
# 
# THIS IS THE ONLY FILE THAT DOESN'T WORK, REST ALL DONE FIXING.
# 
# 
# 
#######

import torch
from ultralytics import YOLO
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

# Load dataset.yaml to get paths and classes
with open('dataset.yaml', 'r') as file:
    dataset_cfg = yaml.safe_load(file)

dataset_path = dataset_cfg['path']
train_dir = dataset_cfg['train']
val_dir = dataset_cfg['val']
train_label_dir = dataset_cfg['train_label_dir']
val_label_dir = dataset_cfg['val_label_dir']
nc = dataset_cfg['nc']  # Number of classes
class_names = dataset_cfg['names']  # List of class labels

# Custom Dataset Class
class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.label_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = int(open(self.label_paths[idx]).read().strip())  # Assuming labels are stored in text files with one label per line

        if self.transform:
            image = self.transform(image)

        return image, label

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to 640x640 for YOLOv8 input
    transforms.ToTensor(),          # Convert image to tensor
])

# Load the dataset using the custom class
train_dataset = ChestXRayDataset(
    os.path.join(dataset_path, train_dir),
    os.path.join(dataset_path, train_label_dir),
    transform=transform
)
val_dataset = ChestXRayDataset(
    os.path.join(dataset_path, val_dir),
    os.path.join(dataset_path, val_label_dir),
    transform=transform
)

# Create DataLoaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the YOLOv8 Pretrained Model
model = YOLO('yolov8x.pt')  # You can choose a smaller version like yolov8n.pt for faster training

# Modify the model's classification head
# Set the number of classes (nc) and change the head output layer
model.model[-1] = torch.nn.Sequential(
    torch.nn.Conv2d(160, nc, kernel_size=(1, 1), stride=1, padding=0),  # Update the output layer
    torch.nn.Flatten(),
    torch.nn.Softmax(dim=1)  # Ensure the output is a probability distribution
)

#  Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss Function and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass through YOLOv8
        outputs = model(inputs)

        # Get predictions from YOLOv8
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Convert YOLOv8 predictions to class predictions
        pred_classes = torch.argmax(outputs.pred[0][:, 5:], dim=1)
        correct_predictions += (pred_classes == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluate the Model
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Get the predicted classes
        pred_classes = torch.argmax(outputs.pred[0][:, 5:], dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_classes.cpu().numpy())

# Compute accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

# Optionally, Save the Model
torch.save(model.state_dict(), "chest_xray_yolov8_model.pth")

# Visualize Some Predictions
def visualize_predictions(inputs, labels, predictions):
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axes[i].imshow(inputs[i].transpose(1, 2, 0))
        axes[i].set_title(f"True: {labels[i]}, Pred: {predictions[i]}")
        axes[i].axis("off")
    plt.show()

# Show some predictions
visualize_predictions(inputs, labels, torch.tensor(all_preds))
