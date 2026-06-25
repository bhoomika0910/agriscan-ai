"""Streamlit frontend for AgriScan AI."""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL = f"{API_URL.rstrip('/')}/predict"
TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "60"))


def _page_config() -> None:
    """Configure Streamlit page metadata and layout."""
    st.set_page_config(page_title="AgriScan AI", page_icon="🌿", layout="wide")


def _render_header() -> None:
    """Render application heading and short user guidance."""
    st.title("🌿 AgriScan AI")
    st.caption(
        "Upload or capture a crop leaf photo to detect disease/pests and get treatment advice."
    )


def _get_image_bytes() -> tuple[bytes | None, str | None]:
    """Return uploaded image bytes and a filename if available."""
    upload_col, camera_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

    with camera_col:
        camera_file = st.camera_input("Capture leaf image")

    selected_file = uploaded_file or camera_file
    if selected_file is None:
        return None, None

    return selected_file.getvalue(), selected_file.name or "captured_leaf.jpg"


def _predict(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """Call backend predict endpoint and return parsed JSON response."""
    mime_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    files = {"file": (filename, io.BytesIO(image_bytes), mime_type)}

    response = requests.post(PREDICT_URL, files=files, timeout=TIMEOUT_SECONDS)

    if response.status_code != 200:
        try:
            details = response.json().get("detail", "Unexpected server error")
        except ValueError:
            details = response.text or "Unexpected server error"
        raise RuntimeError(str(details))

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Backend returned invalid JSON.") from exc


def _render_results(result: dict[str, Any], image_bytes: bytes) -> None:
    """Render prediction output in a clear multi-column layout."""
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Uploaded Image")
        st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)

    with right_col:
        disease = str(result.get("disease", "Unknown"))
        confidence = float(result.get("confidence", 0.0))

        st.subheader("Detection Result")
        st.markdown(f"**Disease / Pest:** {disease}")
        st.progress(max(0.0, min(1.0, confidence)))
        st.caption(f"Confidence: {confidence * 100:.2f}%")

        st.subheader("Recommended Remedy")
        st.write(str(result.get("remedy", "No remedy information available.")))

        st.subheader("Dosage")
        st.write(str(result.get("dosage", "No dosage information available.")))

        st.subheader("Prevention")
        st.write(str(result.get("prevention", "No prevention information available.")))


def main() -> None:
    """Run the Streamlit application."""
    _page_config()
    _render_header()

    image_bytes, filename = _get_image_bytes()
    if image_bytes is None or filename is None:
        st.info("Upload or capture a leaf image to begin analysis.")
        return

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Running disease detection..."):
            try:
                result = _predict(image_bytes=image_bytes, filename=filename)
            except requests.RequestException:
                st.error(
                    "Could not reach the AgriScan API. Check backend status and API_URL setting."
                )
                return
            except RuntimeError as exc:
                st.error(f"Prediction failed: {exc}")
                return

        _render_results(result=result, image_bytes=image_bytes)


if __name__ == "__main__":
    main()
