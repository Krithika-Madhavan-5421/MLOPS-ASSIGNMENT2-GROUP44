import os
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T
import logging
import time
from collections import defaultdict

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



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

request_metrics = {
    "total_requests": 0,
    "total_latency": 0.0
}


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

