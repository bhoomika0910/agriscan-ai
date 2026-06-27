"""FastAPI entry point for AgriScan AI backend."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

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
    ai_analysis: Optional[str] = None


def _load_dotenv_manually() -> None:
    """Manually parse and load .env variables into os.environ for local dev."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        try:
            with env_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        val_clean = val.strip().strip("'\"")
                        os.environ[key.strip()] = val_clean
        except Exception as e:
            print(f"Warning: Failed to load .env manually: {e}")


_load_dotenv_manually()


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
    """Predict root/soil condition from an uploaded image and return treatment advice."""
    api_key_check = os.getenv("GEMINI_API_KEY")
    try:
        with open("app_debug.log", "a") as f:
            f.write(f"[Predict] Called. API Key loaded: {'Yes' if api_key_check else 'No'}\n")
    except Exception:
        pass
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

    # Validate confidence threshold
    if prediction.confidence < 0.5:
        raise HTTPException(
            status_code=400,
            detail=f"Low confidence prediction ({prediction.confidence:.1%}). "
            "Please upload a clearer image of the affected root or soil.",
        )

    # Check if disease is in knowledge base
    from app.treatment import _load_knowledge_base
    kb = _load_knowledge_base()
    if prediction.disease not in kb:
        raise HTTPException(
            status_code=400,
            detail=f"'{prediction.disease}' is not a recognized root or soil condition. "
            "Ensure you're uploading a crop root or soil sample image.",
        )

    # Query Gemini for rich dynamic recommendations if key is configured
    api_key = os.getenv("GEMINI_API_KEY")
    ai_analysis = None
    if api_key and api_key.strip():
        try:
            system_prompt = (
                "You are an expert agronomist and soil scientist advisor for AgriScan UnderGround.\n"
                "Your task is to analyze the diagnosed root or soil condition and return a detailed, professional, "
                "actionable health report in Markdown format. Address the farmer directly and cover these sections:\n"
                "1. **Condition Overview**: Explain why this happens, symptoms, and potential impact on crop yield.\n"
                "2. **Quick Remedies**: Focus on eco-friendly, organic remedies (compost, green manures, bio-controls).\n"
                "3. **Chemical treatments**: If necessary, suggest safe chemical treatments with specific safety warnings.\n"
                "4. **Long-Term Preventive Soil Care**: Detail crop rotation, soil aeration, drainage improvements, or soil amendments.\n"
                "Rules:\n"
                "1. Use simple, supportive, and direct language.\n"
                "2. Keep the advice tailored to the specific diagnosed class.\n"
                "3. Format exactly as clean Markdown."
            )
            user_message = f"Diagnosed Soil/Root Condition: {prediction.disease} (Confidence: {prediction.confidence:.1%})"
            gemini_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_message}]
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": system_prompt}]
                }
            }
            import requests as sync_requests
            # Use gemini-flash-latest as supported by the API key
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
            resp = sync_requests.post(url, json=gemini_payload, headers={"Content-Type": "application/json"}, timeout=15.0)
            if resp.status_code == 200:
                resp_json = resp.json()
                candidates = resp_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        ai_analysis = parts[0].get("text", "")
            else:
                with open("app_debug.log", "a") as f:
                    f.write(f"Gemini API request failed with status code {resp.status_code}: {resp.text}\n")
        except Exception as e:
            with open("app_debug.log", "a") as f:
                f.write(f"Error querying Gemini API: {e}\n")

    return PredictionResponse(
        disease=prediction.disease,
        confidence=prediction.confidence,
        remedy=treatment.remedy,
        dosage=treatment.dosage,
        prevention=treatment.prevention,
        ai_analysis=ai_analysis,
    )
