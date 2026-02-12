import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T

# ---------- model definition (same as training) ----------

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

# ---------- load model ----------

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_cnn.pt")

device = torch.device("cpu")

model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- preprocessing ----------

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# ---------- FastAPI app ----------

app = FastAPI(title="Cats vs Dogs Inference API")

# health check
@app.get("/health")
def health():
    return {"status": "ok"}

# prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "dog" if prob >= 0.5 else "cat"

    return {
        "probability_dog": prob,
        "predicted_label": label
    }
