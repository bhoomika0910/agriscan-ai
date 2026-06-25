"""Treatment lookup utilities for AgriScan AI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class TreatmentLookupError(Exception):
    """Raised when treatment knowledge base cannot be loaded."""


@dataclass(frozen=True)
class TreatmentInfo:
    """Structured treatment recommendation details."""

    remedy: str
    dosage: str
    prevention: str


DEFAULT_TREATMENT = TreatmentInfo(
    remedy="Disease not confidently mapped. Consult an agronomist for field-specific guidance.",
    dosage="Follow local extension officer advice before applying any agrochemical.",
    prevention="Isolate affected plants, monitor spread, and maintain field sanitation.",
)


def _resolve_knowledge_base_path() -> Path:
    """Resolve knowledge base file path from environment."""
    default_path = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"
    return Path(os.getenv("KNOWLEDGE_BASE_PATH", str(default_path))).expanduser().resolve()


@lru_cache(maxsize=1)
def _load_knowledge_base() -> dict[str, dict[str, Any]]:
    """Load and cache disease-to-treatment mapping from JSON."""
    kb_path = _resolve_knowledge_base_path()

    if not kb_path.exists():
        raise TreatmentLookupError(f"Knowledge base file not found: {kb_path}")

    try:
        with kb_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise TreatmentLookupError(f"Invalid knowledge base JSON format: {exc}") from exc
    except OSError as exc:
        raise TreatmentLookupError(f"Unable to read knowledge base file: {exc}") from exc

    if not isinstance(payload, dict):
        raise TreatmentLookupError("Knowledge base content must be a JSON object.")

    return payload


def get_treatment_for_disease(disease: str) -> TreatmentInfo:
    """Return treatment details for a disease class or safe default if unknown."""
    knowledge_base = _load_knowledge_base()
    entry = knowledge_base.get(disease)

    if not isinstance(entry, dict):
        return DEFAULT_TREATMENT

    remedy = str(entry.get("remedy", DEFAULT_TREATMENT.remedy))
    dosage = str(entry.get("dosage", DEFAULT_TREATMENT.dosage))
    prevention = str(entry.get("prevention", DEFAULT_TREATMENT.prevention))

    return TreatmentInfo(remedy=remedy, dosage=dosage, prevention=prevention)
