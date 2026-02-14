import requests
import os
from sklearn.metrics import accuracy_score
import sys

API_URL = "http://localhost:8000/predict"
TEST_FOLDER = "data/tests"

true_labels = []
pred_labels = []

for filename in os.listdir(TEST_FOLDER):

    if filename.lower().endswith(".jpg"):

        # assumes filename contains class name
        label = "dog" if "dog" in filename.lower() else "cat"
        true_labels.append(label)

        with open(os.path.join(TEST_FOLDER, filename), "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        if response.status_code != 200:
            print("Prediction request failed")
            sys.exit(1)

        pred = response.json()["predicted_label"]
        pred_labels.append(pred)

accuracy = accuracy_score(true_labels, pred_labels)

print("Post-deployment accuracy:", accuracy)

# Fail pipeline if accuracy too low
THRESHOLD = 0.6

if accuracy < THRESHOLD:
    print("Accuracy below threshold!")
    sys.exit(1)
