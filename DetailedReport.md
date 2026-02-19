
---

# Cats vs Dogs Classification

## End-to-End Production-Grade MLOps Pipeline

---

## Team Details

| S.No | Register Number | Name                              | Contribution |
| ---- | --------------- | --------------------------------- | ------------ |
| 1    | 2024aa05421     | Krithika Madhavan                 | 100%         |
| 2    | 2024aa05435     | Yarragondla Rugmangadha Reddy     | 100%         |
| 3    | 2024aa05423     | Payel Karmakar                    | 100%         |
| 4    | 2024aa05870     | Deepak Sindhu                     | 100%         |
| 5    | 2024ab05227     | Parab Prathamesh Prafulla Pradnya | 100%         |

---

# 1️. Project Overview

This project implements a **complete production-ready MLOps pipeline** for binary image classification (Cats vs Dogs).

The pipeline demonstrates:

* Reproducible data versioning (DVC)
* Experiment tracking (MLflow)
* Model training and artifact logging
* Containerized inference (FastAPI + Docker)
* CI/CD automation (GitHub Actions)
* Deployment validation
* Monitoring & post-deployment evaluation

Dataset:

Kaggle Cats vs Dogs Dataset

Images are:

* Converted to RGB
* Resized to 224×224
* Split into train/val/test (80/10/10)
* Augmented for generalization

---

# 2️. System Architecture

```
Raw Dataset
      ↓
DVC Data Pipeline
      ↓
Training (MLflow Tracking)
      ↓
Model Artifact
      ↓
Docker Build (CI)
      ↓
Docker Hub
      ↓
CD Deployment (Docker Compose)
      ↓
FastAPI Service
      ↓
Monitoring & Evaluation
```

---

# 3️. Setup Instructions

## 3.1 Requirements

* Python 3.10+
* Git
* Docker
* DVC
* MLflow

---

## 3.2 Clone Repository

```bash
git clone https://github.com/Krithika-Madhavan-5421/MLOPS-ASSIGNMENT2-GROUP44.git
cd MLOPS-ASSIGNMENT2-GROUP44
```

---

## 3.3 Create Virtual Environment

```bash
conda create -n cats-dogs-mlops python=3.10 -y
conda activate cats-dogs-mlops
```

or

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3.4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4️. Data Versioning & Reproducibility (DVC)

## 4.1 DVC Pipeline Structure

The project defines reproducible stages in `dvc.yaml`:

```yaml
stages:
  data_preparation:
    cmd: python src/data_preparation.py
    deps:
      - src/data_preparation.py
      - data/raw
    outs:
      - data/processed

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed
    outs:
      - models/model.pth
```

---

## 4.2 Running Full Pipeline

To reproduce the entire pipeline:

```bash
dvc repro
```

This will:

* Run data preprocessing
* Train model
* Regenerate artifacts if dependencies changed

---

## 4.3 Pulling Artifacts from Remote

If DVC remote is configured:

```bash
dvc pull
```

Restores:

* Processed dataset
* Model weights

---

## Why DVC?

* Large data not stored in Git
* Reproducible training
* Controlled dataset versioning
* Automatic dependency tracking

---

# 5️. Model Development & Experiment Tracking

## 5.1 Baseline CNN Model

Architecture:

* 3 Conv blocks
* Batch Normalization
* ReLU
* MaxPooling
* Global Average Pooling
* Fully connected output

Loss Function:

```
BCEWithLogitsLoss
```

---

## 5.2 Training

```bash
python src/train.py
```

Includes:

* Data augmentation
* Learning rate scheduler
* Weight decay
* MLflow experiment logging

---

## 5.3 MLflow Tracking

MLflow logs:

* Hyperparameters
* Loss curves
* Validation accuracy
* Confusion matrix
* Model artifacts

Launch UI:

```bash
mlflow ui
```

Access:

```
http://localhost:5000
```

![MLFlow1.jpg](screenshots/MLFlow1.jpg)
![MLFlow2.jpg](screenshots/MLFlow2.jpg)
![MLFlow3.jpg](screenshots/MLFlow3.jpg)
![MLFlow4.jpg](screenshots/MLFlow4.jpg)
---

# 6️. Model Serving & Dockerization

## 6.1 FastAPI Inference Service

Endpoints:

| Endpoint   | Purpose              |
| ---------- | -------------------- |
| `/health`  | Health check         |
| `/predict` | Image classification |
| `/metrics` | Runtime metrics      |

Run locally:

```bash
uvicorn src.inference_app:app --host 0.0.0.0 --port 8000
```

Swagger:

```
http://localhost:8000/docs
```

![Swagger1.jpg](screenshots/Swagger1.jpg)

---

## 6.2 Docker Build

```bash
docker build -t <dockerhub-username>/cats-dogs-inference:latest .
```

Run:

```bash
docker run -p 8000:8000 <dockerhub-username>/cats-dogs-inference:latest
```

---

# 7️. Docker Hub Integration

## Manual Push

```bash
docker login
docker push <dockerhub-username>/cats-dogs-inference:latest
```
![Dockerhub1.jpg](screenshots/Dockerhub1.jpg)
---

## GitHub Secrets Required

Add in repository settings:

* `DOCKER_USERNAME`
* `DOCKER_PASSWORD`

---

## CI Docker Push Workflow

```yaml
- name: Login to Docker Hub
  run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

- name: Build Image
  run: docker build -t ${{ secrets.DOCKER_USERNAME }}/cats-dogs-inference:latest .

- name: Push Image
  run: docker push ${{ secrets.DOCKER_USERNAME }}/cats-dogs-inference:latest
```

---

# 8️. Continuous Integration (CI)

Triggered on push.

Pipeline:

1. Install dependencies
2. Run pytest unit tests
3. Build Docker image
4. Push image to Docker Hub

![ci1.jpg](screenshots/ci1.jpg)
---

# 9️. Continuous Deployment (CD)

Deployment via Docker Compose.

Example:

```yaml
version: "3.8"
services:
  inference:
    image: <dockerhub-username>/cats-dogs-inference:latest
    ports:
      - "8000:8000"
    restart: always
```

Deploy:

```bash
docker compose pull
docker compose up -d
```

Smoke tests:

* `/health`
* `/predict`

Fail deployment if tests fail.

![cd1.jpg](screenshots/cd1.jpg)
---

# 10. Monitoring & Post-Deployment Evaluation

## Runtime Metrics

`GET /metrics` returns:

* Total requests
* Average latency
* Prediction count

---

## Post-Deployment Accuracy Validation

Separate workflow:

* Sends labeled test data
* Computes accuracy
* Enforces threshold
* Fails if below threshold

Example output:

```
Post-deployment accuracy: 1.0
```

![model-monitoring1.jpg](screenshots/model-monitoring1.jpg)
---

# 1️1️. Engineering Decisions

* Global Average Pooling reduces parameters
* DVC ensures reproducibility
* MLflow enables experiment tracking
* Docker ensures portability
* CI/CD ensures automation
* Lazy model loading for performance
* Separate monitoring workflow

---

# 1️2️. Repository

🔗
[https://github.com/Krithika-Madhavan-5421/MLOPS-ASSIGNMENT2-GROUP44](https://github.com/Krithika-Madhavan-5421/MLOPS-ASSIGNMENT2-GROUP44)

---

# 1️3️. Conclusion

This project demonstrates a complete industrial-grade ML lifecycle including:

* Reproducible training
* Experiment management
* Containerized inference
* Automated CI/CD
* Monitoring & validation

It reflects production MLOps best practices and ensures:

* Scalability
* Automation
* Reliability
* Deployment readiness
* Performance validation

---