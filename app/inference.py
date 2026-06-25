"""Inference utilities for AgriScan AI."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class InferenceError(Exception):
    """Raised when model loading or prediction fails."""


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction result returned by the inference layer."""

    disease: str
    confidence: float


def _resolve_model_path() -> Path:
    """Resolve model path from environment with a project-relative default."""
    default_model = Path(__file__).resolve().parent / "models" / "best.pt"
    return Path(os.getenv("MODEL_PATH", str(default_model))).expanduser().resolve()


@lru_cache(maxsize=1)
def _load_model() -> YOLO:
    """Load the YOLO model with graceful fallback to pretrained weights."""
    model_path = _resolve_model_path()

    if model_path.exists():
        try:
            return YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover
            raise InferenceError(f"Failed to load model from {model_path}: {exc}") from exc

    warnings.warn(
        (
            f"Model file not found at {model_path}. "
            "Falling back to pretrained yolov8n.pt weights."
        ),
        RuntimeWarning,
        stacklevel=2,
    )

    try:
        return YOLO("yolov8n.pt")
    except Exception as exc:  # pragma: no cover
        raise InferenceError(
            "No local model found and fallback pretrained weights could not be loaded."
        ) from exc


def _class_name(names: Any, class_idx: int) -> str:
    """Resolve a class index to a readable class name."""
    if isinstance(names, dict):
        return str(names.get(class_idx, f"class_{class_idx}"))
    if isinstance(names, list) and 0 <= class_idx < len(names):
        return str(names[class_idx])
    return f"class_{class_idx}"


def predict_disease(image_rgb: np.ndarray) -> PredictionResult:
    """Run inference and return the best disease class and confidence score."""
    model = _load_model()

    try:
        results = model.predict(source=image_rgb, verbose=False)
    except Exception as exc:  # pragma: no cover
        raise InferenceError(f"Prediction failed: {exc}") from exc

    if not results:
        raise InferenceError("No prediction result was returned by the model.")

    result = results[0]
    names = getattr(result, "names", getattr(model, "names", {}))

    probs = getattr(result, "probs", None)
    if probs is not None and getattr(probs, "data", None) is not None:
        probs_arr = probs.data.detach().cpu().numpy()
        class_idx = int(np.argmax(probs_arr))
        confidence = float(probs_arr[class_idx])
        return PredictionResult(
            disease=_class_name(names, class_idx),
            confidence=confidence,
        )

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        class_idx = int(boxes.cls[0].item())
        confidence = float(boxes.conf[0].item())
        return PredictionResult(
            disease=_class_name(names, class_idx),
            confidence=confidence,
        )

    raise InferenceError("Model returned no recognizable classes for the input image.")
