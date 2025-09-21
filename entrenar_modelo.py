import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------
# CONFIGURACIÓN GENERAL
# ----------------------

train_dir = "C:/Users/Natilla/Documents/MachineLearningConfiabilidad/train_frames"
test_dir = "C:/Users/Natilla/Documents/MachineLearningConfiabilidad/test_frames"

batch_size = 16
epochs = 10
lr = 0.001
image_size = 64

# ----------------------
# TRANSFORMACIONES
# ----------------------

transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# ----------------------
# DATASETS Y DATALOADERS
# ----------------------

train_data = datasets.ImageFolder(train_dir, transform=transform_train)
test_data = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ----------------------
# DEFINICIÓN DEL MODELO
# ----------------------

class RedConfiabilidad(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ----------------------
# ENTRENAMIENTO DEL MODELO
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RedConfiabilidad(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("⏳ Entrenando modelo...\n")

# Historial
losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    losses.append(running_loss / len(train_loader))
    train_accuracies.append(acc)

    # Evaluación en test set
    model.eval()
    correct_test = total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100 * correct_test / total_test
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {losses[-1]:.4f} - Train Acc: {acc:.2f}% - Test Acc: {test_acc:.2f}%")

# ----------------------
# GUARDAR EL MODELO
# ----------------------

torch.save(model.state_dict(), "modelo_confiabilidad.pth")
print("\n💾 Modelo guardado como 'modelo_confiabilidad.pth'")

# ----------------------
# GUARDAR HISTORIAL CSV
# ----------------------

with open("historial_entrenamiento.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss", "Train Accuracy", "Test Accuracy"])
    for i in range(epochs):
        writer.writerow([i+1, losses[i], train_accuracies[i], test_accuracies[i]])

print("📈 Historial guardado como 'historial_entrenamiento.csv'")

from collections import Counter

print(Counter(train_labels))  # durante entrenamiento
