import requests
import os
import sys
from sklearn.metrics import accuracy_score

API_URL = "http://localhost:8000/predict"
TEST_ROOT = "data/tests"
THRESHOLD = 0.6   # Fail monitoring if below this


true_labels = []
pred_labels = []

# Loop through class folders
for class_name in ["cats", "dogs"]:

    class_folder = os.path.join(TEST_ROOT, class_name)

    if not os.path.exists(class_folder):
        print(f"Missing folder: {class_folder}")
        sys.exit(1)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            true_label = "cat" if class_name == "cats" else "dog"
            true_labels.append(true_label)

            file_path = os.path.join(class_folder, filename)

            try:
                with open(file_path, "rb") as f:
                    response = requests.post(API_URL, files={"file": f})

                if response.status_code != 200:
                    print(f"Request failed for {filename}")
                    sys.exit(1)

                pred = response.json()["predicted_label"]
                pred_labels.append(pred)

                print(f"{filename} | True: {true_label} | Pred: {pred}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                sys.exit(1)


if len(true_labels) == 0:
    print("No test images found.")
    sys.exit(1)

accuracy = accuracy_score(true_labels, pred_labels)

print("\nPost-deployment accuracy:", accuracy)

# Fail monitoring pipeline if performance drops
if accuracy < THRESHOLD:
    print("Accuracy below threshold!")
    sys.exit(1)

print("Monitoring check passed.")
