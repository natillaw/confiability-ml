import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------
# CONFIGURACIÓN GENERAL
# ----------------------

train_dir = r"C:\Users\Natilla\Music\MACHINE LEARNING\data\train_frames"
test_dir  = r"C:\Users\Natilla\Music\MACHINE LEARNING\data\test_frames"

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
# VALIDACIÓN DE DATOS
# ----------------------

for clase in ["lie", "truth"]:
    clase_test_dir = os.path.join(test_dir, clase)
    if not os.path.exists(clase_test_dir) or not any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(clase_test_dir)):
        raise FileNotFoundError(f"No se encontraron imágenes válidas en: {clase_test_dir}")

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
# ENTRENAMIENTO
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RedConfiabilidad(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("⏳ Entrenando modelo...\n")

# Historial
losses = []
train_metrics = []
test_metrics = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    y_true_train, y_pred_train = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_acc = accuracy_score(y_true_train, y_pred_train) * 100
    train_prec = precision_score(y_true_train, y_pred_train, zero_division=0)
    train_recall = recall_score(y_true_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)
    losses.append(running_loss / len(train_loader))
    train_metrics.append((train_acc, train_prec, train_recall, train_f1))

    # Evaluación
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())

    test_acc = accuracy_score(y_true_test, y_pred_test) * 100
    test_prec = precision_score(y_true_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_true_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
    test_metrics.append((test_acc, test_prec, test_recall, test_f1))

    print(f"📦 Epoch {epoch+1}/{epochs} | Loss: {losses[-1]:.4f}")
    print(f"  🟢 Train  Acc: {train_acc:.2f}% | Prec: {train_prec:.2f} | Rec: {train_recall:.2f} | F1: {train_f1:.2f}")
    print(f"  🔵 Test   Acc: {test_acc:.2f}% | Prec: {test_prec:.2f} | Rec: {test_recall:.2f} | F1: {test_f1:.2f}\n")

# ----------------------
# GUARDAR MODELO Y CSV
# ----------------------

save_base = r"C:\Users\Natilla\Music\MACHINE LEARNING\data"
model_path = os.path.join(save_base, "modelo_confiabilidad.pth")
torch.save(model.state_dict(), model_path)
print(f"💾 Modelo guardado en: {model_path}")

csv_path = os.path.join(save_base, "historial_entrenamiento.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss", "Train Acc", "Train Prec", "Train Recall", "Train F1",
                     "Test Acc", "Test Prec", "Test Recall", "Test F1"])
    for i in range(epochs):
        writer.writerow([
            i+1, losses[i],
            *train_metrics[i],
            *test_metrics[i]
        ])
print(f"📈 Historial guardado en: {csv_path}")
