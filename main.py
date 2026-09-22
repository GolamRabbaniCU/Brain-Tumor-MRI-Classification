"""
main.py

FastAPI app exposing /health and /predict endpoints for the
Brain Tumor MRI Classifier. Runs locally on CPU; loads predictor.py
once at startup.
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import predictor

app = FastAPI(title="Brain Tumor MRI Classifier")


class PredictionOutput(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ViT fine-tuned on Brain Tumor MRI",
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predictor.predict(image_bytes)
    return PredictionOutput(**result)
