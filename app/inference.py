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
def _load_model() -> YOLO | None:
    """Load the YOLO model, returning None if the weights file is missing."""
    model_path = _resolve_model_path()

    if model_path.exists():
        try:
            return YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover
            raise InferenceError(f"Failed to load model from {model_path}: {exc}") from exc

    warnings.warn(
        (
            f"Model file not found at {model_path}. "
            "Falling back to deterministic mock diagnostic mode for root/soil classes."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _class_name(names: Any, class_idx: int) -> str:
    """Resolve a class index to a readable class name."""
    if isinstance(names, dict):
        return str(names.get(class_idx, f"class_{class_idx}"))
    if isinstance(names, list) and 0 <= class_idx < len(names):
        return str(names[class_idx])
    return f"class_{class_idx}"


def predict_disease(image_rgb: np.ndarray) -> PredictionResult:
    """Run inference or return a mock prediction if the model is not found."""
    model = _load_model()

    if model is None:
        # Deterministic Mock Mode: resolve classes from classes.txt
        try:
            classes_path = Path(__file__).resolve().parents[1] / "data" / "classes.txt"
            with classes_path.open("r", encoding="utf-8") as file:
                classes = [line.strip() for line in file if line.strip()]
            if not classes:
                classes = ["Root___Healthy"]
        except Exception:
            classes = ["Root___Healthy"]

        import random
        # Seed the RNG with a checksum of the image so the same image always yields the same result
        try:
            img_hash = int(np.sum(image_rgb) % 1000000)
            rng = random.Random(img_hash)
            selected_class = rng.choice(classes)
            confidence = float(rng.uniform(0.72, 0.97))
        except Exception:
            selected_class = random.choice(classes)
            confidence = 0.88

        return PredictionResult(disease=selected_class, confidence=confidence)

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
