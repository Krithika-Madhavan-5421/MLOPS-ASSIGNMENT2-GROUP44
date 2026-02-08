import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

DATA_DIR = "data/processed"

transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor()
])

transform_val = T.Compose([
    T.ToTensor()
])

train_ds = ImageFolder(f"{DATA_DIR}/train", transform=transform_train)
val_ds   = ImageFolder(f"{DATA_DIR}/val", transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mlflow.set_experiment("cats_vs_dogs_baseline")

with mlflow.start_run():

    mlflow.log_param("batch_size", 32)
    mlflow.log_param("lr", 1e-3)
    mlflow.log_param("model", "SimpleCNN")

    num_epochs = 2   # keep small for M1

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs} started...")
        start_time = time.time()

        # -------- training --------
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.float().to(device)

            optimizer.zero_grad()
            out = model(x).squeeze()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f}"
                )

        # -------- validation --------
        model.eval()
        correct, total = 0, 0
        all_preds, all_gt = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze()
                preds = (torch.sigmoid(out) > 0.5).long()

                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_gt.extend(y.cpu().numpy())

        acc = correct / total
        avg_loss = train_loss / len(train_loader)

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_accuracy", acc, step=epoch)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} finished | "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

    # -------- artifacts --------
    cm = confusion_matrix(all_gt, all_preds)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/baseline_cnn.pt")
    mlflow.log_artifact("models/baseline_cnn.pt")

    torch.save(model.state_dict(), "models/baseline_cnn.pt")
    mlflow.log_artifact("models/baseline_cnn.pt")

    mlflow.pytorch.log_model(model, "model")
