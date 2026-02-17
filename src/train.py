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

# ----------------------------
# Faster Transforms (128x128)
# ----------------------------

transform_train = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_ds = ImageFolder(f"{DATA_DIR}/train", transform=transform_train)
val_ds   = ImageFolder(f"{DATA_DIR}/val", transform=transform_val)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    num_workers=2
)

# ----------------------------
# Small Efficient CNN
# ----------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleCNN().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5
)

# ----------------------------
# MLflow Setup
# ----------------------------

mlflow.set_experiment("cats_vs_dogs_fast_cpu")

with mlflow.start_run():

    mlflow.log_param("image_size", 128)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("lr", 1e-4)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("weight_decay", 1e-4)

    num_epochs = 5

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()

        # -------- Training --------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)

            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # -------- Validation --------
        model.eval()
        correct, total = 0, 0
        all_preds, all_gt = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).long()

                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_gt.extend(y.cpu().numpy())

        acc = correct / total
        avg_loss = train_loss / len(train_loader)

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_accuracy", acc, step=epoch)

        scheduler.step()

        print(
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Time: {time.time() - start_time:.1f}s"
        )

    # -------- Confusion Matrix --------
    cm = confusion_matrix(all_gt, all_preds)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # -------- Save Model --------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_cnn.pt")
    mlflow.log_artifact("models/baseline_cnn.pt")

    mlflow.pytorch.log_model(model, "model")

print("Training complete.")
