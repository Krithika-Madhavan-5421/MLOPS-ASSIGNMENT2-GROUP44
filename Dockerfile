FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference_app.py ./src/inference_app.py
COPY models/baseline_cnn.pt ./models/baseline_cnn.pt

EXPOSE 8000

CMD ["uvicorn", "src.inference_app:app", "--host", "0.0.0.0", "--port", "8000"]
