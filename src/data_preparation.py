import os
import cv2
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
IMG_SIZE = 224

def prepare():

    images = []
    labels = []

    for label, cls in enumerate(["cats", "dogs"]):
        folder = Path(RAW_DIR) / cls
        for img_path in folder.glob("*"):
            images.append(str(img_path))
            labels.append(label)

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls, test_size=0.5, stratify=temp_lbls, random_state=42
    )

    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls)
    }

    for split in splits:
        for cls in ["cats", "dogs"]:
            os.makedirs(f"{OUT_DIR}/{split}/{cls}", exist_ok=True)

    for split, (imgs, lbls) in splits.items():
        for p, l in zip(imgs, lbls):

            img = cv2.imread(p)

            if img is None:
                continue   # skip unreadable / corrupted images

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            cls = "cats" if l == 0 else "dogs"
            name = Path(p).name

            cv2.imwrite(
                f"{OUT_DIR}/{split}/{cls}/{name}",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

if __name__ == "__main__":
    prepare()
