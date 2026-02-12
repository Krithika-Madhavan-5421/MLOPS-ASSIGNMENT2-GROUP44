import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T


# ----------------------------
# Model definition
# ----------------------------

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


# ----------------------------
# Lazy model loading
# ----------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_cnn.pt")
device = torch.device("cpu")

_model = None


def get_model():
    global _model

    if _model is None:
        m = SimpleCNN()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        _model = m

    return _model


# ----------------------------
# Prediction utility
# ----------------------------

def predict_tensor(model, x):
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "dog" if prob >= 0.5 else "cat"
    return prob, label


# ----------------------------
# Preprocessing
# ----------------------------

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Cats vs Dogs Inference API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0)

    model = get_model()
    prob, label = predict_tensor(model, x)

    return {
        "probability_dog": prob,
        "predicted_label": label
    }
