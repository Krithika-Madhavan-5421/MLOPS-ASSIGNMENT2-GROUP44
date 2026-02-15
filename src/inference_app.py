import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T
import logging
import time

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ----------------------------
# Small Improved SimpleCNN (MATCHES TRAIN.PY)
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

        # 🔥 Global Average Pooling
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

# ----------------------------
# Lazy Model Loading
# ----------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_cnn.pt")
device = torch.device("cpu")

_model = None

def get_model():
    global _model

    if _model is None:
        model = SimpleCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        _model = model

    return _model

# ----------------------------
# Monitoring Metrics
# ----------------------------

request_metrics = {
    "total_requests": 0,
    "total_latency": 0.0
}

# ----------------------------
# Prediction Utility
# ----------------------------

def predict_tensor(model, x):
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "dog" if prob >= 0.5 else "cat"
    return prob, label

# ----------------------------
# Preprocessing (MUST match train.py)
# ----------------------------

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Cats vs Dogs Inference API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.time()
    request_metrics["total_requests"] += 1

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = transform(image).unsqueeze(0)

        model = get_model()
        prob, label = predict_tensor(model, x)

        latency = time.time() - start_time
        request_metrics["total_latency"] += latency

        logger.info(
            f"Prediction made | label={label} | prob={prob:.4f} | latency={latency:.4f}s"
        )

        return {
            "probability_dog": prob,
            "predicted_label": label,
            "latency_seconds": latency
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

@app.get("/metrics")
def metrics():

    total = request_metrics["total_requests"]

    avg_latency = (
        request_metrics["total_latency"] / total
        if total > 0 else 0
    )

    return {
        "total_requests": total,
        "average_latency_seconds": avg_latency
    }
