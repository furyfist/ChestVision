from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import LungCancerDataset

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


print("Loading raw dataset from Hugging Face...")
# 1. Load raw dataset from Hugging Face
raw_ds = load_dataset("dorsar/lung-cancer")

print("Creating custom PyTorch datasets...")
# 2. Create instances of our custom dataset for train and validation sets
train_dataset = LungCancerDataset(hg_dataset_split=raw_ds['train'])
val_dataset = LungCancerDataset(hg_dataset_split=raw_ds['validation'])

print("Creating DataLoaders...")
# 3. Wrap the datasets in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Check if it works
print(f"Successfully created a train loader with {len(train_loader)} batches of size 32.")
print(f"Successfully created a validation loader with {len(val_loader)} batches of size 32.")

# Model setup (using transfer learning)
print("Setting up the model....")

# loding pre-trained model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freezing the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# replace the final layer
# The original model was for 1000 classes (ImageNet). We need 7 for our dataset.
num_features = model.fc.in_features # Get the number of input features for the final layer
model.fc = nn.Linear(num_features, 7) # Replace it with a new layer for our 7 classes

# 4. Move the model to the correct device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded on {device} and final layer replaced for 7 classes.")


# loss function
print("setting up loss function and optimizer.....")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

print("Loss function and optimzer is ready:)")

# Training loop

num_epochs = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Loop over the training data
    for images, labels in train_loader:
        # move data to cpu/gpu
        images = images.to(device)
        labels = labels.to(device)

        # 1. forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # calculate average loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_acc = (correct_predictions / total_predictions) * 100

    # validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val_predictions = 0
    total_val_predictions = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val_predictions += labels.size(0)
            correct_val_predictions += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_acc = (correct_val_predictions / total_val_predictions) * 100

        # statistic of epoch
        # Print statistics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("\n Finished Training")

