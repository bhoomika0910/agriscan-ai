"""FastAPI entry point for AgriScan AI backend."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from app.inference import InferenceError, predict_disease
from app.treatment import TreatmentLookupError, get_treatment_for_disease


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class PredictionResponse(BaseModel):
    """API response schema for disease prediction."""

    disease: str
    confidence: float
    remedy: str
    dosage: str
    prevention: str


app = FastAPI(
    title=os.getenv("API_TITLE", "AgriScan AI API"),
    version="1.0.0",
    description="Crop disease and pest prediction API powered by YOLOv8.",
)

origins_raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
allowed_origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict disease from an uploaded image and return treatment advice."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Use jpg, jpeg, or png.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported content type for image upload.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_rgb = np.array(pil_image)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload.") from exc

    try:
        prediction = predict_disease(image_rgb)
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    try:
        treatment = get_treatment_for_disease(prediction.disease)
    except TreatmentLookupError as exc:
        raise HTTPException(status_code=500, detail=f"Treatment lookup error: {exc}") from exc

    return PredictionResponse(
        disease=prediction.disease,
        confidence=prediction.confidence,
        remedy=treatment.remedy,
        dosage=treatment.dosage,
        prevention=treatment.prevention,
    )
