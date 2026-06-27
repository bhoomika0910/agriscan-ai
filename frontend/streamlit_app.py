"""Streamlit frontend for AgriScan UnderGround."""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8001")
PREDICT_URL = f"{API_URL.rstrip('/')}/predict"
TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "60"))


def _page_config() -> None:
    """Configure Streamlit page metadata and layout."""
    st.set_page_config(
        page_title="AgriScan UnderGround",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def _inject_custom_css() -> None:
    """Inject custom light theme CSS for organic cards, indicators and layouts."""
    st.markdown("""
        <style>
        /* Modern lighter organic container styling */
        .reportview-container {
            background-color: #F7F9F5;
        }
        
        /* Green accent card style */
        .organic-card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(46, 125, 50, 0.04);
            border: 1px solid rgba(46, 125, 50, 0.08);
            margin-bottom: 20px;
        }
        
        .organic-card-title {
            font-size: 14px;
            color: #556B2F;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 16px;
            border-bottom: 2px solid #E8F5E9;
            padding-bottom: 6px;
        }
        
        /* Metric widget custom layout */
        .metric-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 8px 12px;
            background-color: #FAFBF9;
            border-radius: 8px;
            border-left: 4px solid #81C784;
        }
        
        .metric-name {
            font-size: 13px;
            font-weight: 600;
            color: #2E4F32;
        }
        
        .metric-badge {
            font-size: 12px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 6px;
            text-align: right;
        }
        
        /* Badge colors based on status */
        .badge-green { background-color: #E8F5E9; color: #2E7D32; }
        .badge-orange { background-color: #FFF3E0; color: #EF6C00; }
        .badge-red { background-color: #FFEBEE; color: #C62828; }
        .badge-blue { background-color: #E3F2FD; color: #1565C0; }
        </style>
    """, unsafe_allow_html=True)


def _render_header() -> None:
    """Render application heading and short user guidance."""
    st.title("🌿 AgriScan UnderGround")
    st.markdown(
        "**Precision Underground Diagnostics Platform** — Identify root diseases, nutrient deficiencies, "
        "and soil degradation from root or soil sample photos instantly."
    )
    st.markdown("---")


def _get_image_bytes() -> tuple[bytes | None, str | None]:
    """Return uploaded image bytes and a filename if available."""
    st.markdown("""<div class="organic-card">
        <div class="organic-card-title">Provide Root or Soil Sample Image</div>
    """, unsafe_allow_html=True)
    
    input_source = st.radio(
        "Select capture method:", 
        ["📁 Upload File", "📸 Use Camera"], 
        horizontal=True,
        label_visibility="collapsed"
    )
    
    selected_file = None
    if "Upload" in input_source:
        selected_file = st.file_uploader(
            "Upload root system or soil core image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
    else:
        selected_file = st.camera_input("Capture photo using device camera node")

    st.markdown("""</div>""", unsafe_allow_html=True)
    
    if selected_file is None:
        return None, None

    return selected_file.getvalue(), selected_file.name or "captured_soil_root.jpg"


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


def _get_soil_metrics(disease: str) -> dict[str, dict[str, Any]]:
    """Generate mock soil parameter indicators based on predicted disease for visual display."""
    # Default healthy soil status
    metrics = {
        "nitrogen": {"value": "Optimal (135 ppm)", "badge": "badge-green", "progress": 0.72},
        "phosphorus": {"value": "Optimal (48 ppm)", "badge": "badge-green", "progress": 0.68},
        "potassium": {"value": "Optimal (175 ppm)", "badge": "badge-green", "progress": 0.70},
        "moisture": {"value": "Optimal (26%)", "badge": "badge-blue", "progress": 0.52},
        "ph": {"value": "Neutral (6.7)", "badge": "badge-green", "progress": 0.67},
        "microbial": {"value": "Active (85%)", "badge": "badge-green", "progress": 0.85}
    }
    
    if disease == "Root___Rot":
        metrics["moisture"] = {"value": "Saturated (48%)", "badge": "badge-red", "progress": 0.96}
        metrics["microbial"] = {"value": "Stressed (32%)", "badge": "badge-orange", "progress": 0.32}
        metrics["nitrogen"] = {"value": "Leaching (60 ppm)", "badge": "badge-orange", "progress": 0.40}
    elif disease == "Root___Knot_Nematodes":
        metrics["microbial"] = {"value": "Diseased (25%)", "badge": "badge-red", "progress": 0.25}
    elif disease == "Root___Woolly_Aphids":
        metrics["microbial"] = {"value": "Stressed (45%)", "badge": "badge-orange", "progress": 0.45}
    elif disease == "Root___Nitrogen_Deficiency":
        metrics["nitrogen"] = {"value": "Critical (12 ppm)", "badge": "badge-red", "progress": 0.10}
        metrics["microbial"] = {"value": "Sub-optimal (48%)", "badge": "badge-orange", "progress": 0.48}
    elif disease == "Root___Phosphorus_Deficiency":
        metrics["phosphorus"] = {"value": "Critical (6 ppm)", "badge": "badge-red", "progress": 0.08}
    elif disease == "Root___Potassium_Deficiency":
        metrics["potassium"] = {"value": "Critical (35 ppm)", "badge": "badge-red", "progress": 0.14}
    elif disease == "Soil___Compacted":
        metrics["moisture"] = {"value": "Restricted (12%)", "badge": "badge-orange", "progress": 0.24}
        metrics["microbial"] = {"value": "Anaerobic (18%)", "badge": "badge-red", "progress": 0.18}
        metrics["ph"] = {"value": "Alkaline (7.6)", "badge": "badge-orange", "progress": 0.76}
    elif disease == "Soil___Salinity_Crust":
        metrics["ph"] = {"value": "Alkaline (8.5)", "badge": "badge-red", "progress": 0.85}
        metrics["moisture"] = {"value": "Dehydrated (8%)", "badge": "badge-red", "progress": 0.16}
        metrics["microbial"] = {"value": "Inhibited (10%)", "badge": "badge-red", "progress": 0.10}
    elif disease == "Soil___Erosion_Sandy":
        metrics["nitrogen"] = {"value": "Depleted (15 ppm)", "badge": "badge-red", "progress": 0.12}
        metrics["phosphorus"] = {"value": "Depleted (5 ppm)", "badge": "badge-red", "progress": 0.08}
        metrics["potassium"] = {"value": "Depleted (32 ppm)", "badge": "badge-red", "progress": 0.13}
        metrics["moisture"] = {"value": "Draining (9%)", "badge": "badge-red", "progress": 0.18}
        metrics["microbial"] = {"value": "Low (22%)", "badge": "badge-red", "progress": 0.22}

    return metrics


def _render_results(result: dict[str, Any], image_bytes: bytes) -> None:
    """Render prediction output with balanced visual layout and metrics."""
    disease = str(result.get("disease", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    disease_formatted = disease.replace('___', ' — ').replace('_', ' ')

    # 1. Top Banner Card
    st.markdown(f"""
        <div class="organic-card" style="text-align: center; border-left: 6px solid #2E7D32; background: linear-gradient(135deg, #F1F8E9, #E8F5E9); margin-top: 10px;">
            <div style="font-size: 13px; color: #558B2F; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Diagnostic Classification</div>
            <div style="font-size: 28px; font-weight: 800; color: #1B5E20; margin-top: 4px; margin-bottom: 4px;">{disease_formatted}</div>
            <div style="font-size: 16px; font-weight: 600; color: #2E7D32;">Confidence: <span style="font-size: 20px; font-weight: 800;">{confidence * 100:.1f}%</span></div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Split Screen: Image (Left) vs Soil Metrics (Right)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("""<div class="organic-card" style="height: 100%;">
            <div class="organic-card-title">Sample Visualization</div>
        """, unsafe_allow_html=True)
        st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True, caption="Uploaded Root/Soil Core Sample")
        st.markdown("""</div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown("""<div class="organic-card" style="height: 100%;">
            <div class="organic-card-title">Estimated Soil Aggregates & Metrics</div>
        """, unsafe_allow_html=True)
        
        soil_metrics = _get_soil_metrics(disease)
        
        for key, details in soil_metrics.items():
            st.markdown(f"""
                <div class="metric-row">
                    <span class="metric-name">{key.capitalize()} Status</span>
                    <span class="metric-badge {details['badge']}">{details['value']}</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(details["progress"])
            st.markdown("""<div style="margin-bottom: 8px;"></div>""", unsafe_allow_html=True)
            
        st.markdown("""</div>""", unsafe_allow_html=True)

    # Separator
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 3. Full Width Action Tabs below
    tab_rec, tab_ai = st.tabs(["📋 Recommended Actions", "🤖 AI Agronomist Report"])

    with tab_rec:
        st.markdown("""<div class="organic-card">""", unsafe_allow_html=True)
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("### 💊 Remedy & Recovery Plan")
            st.write(str(result.get("remedy", "No remedy details available.")))
            st.markdown("---")
            st.markdown("### ⚖️ Application & Dosage")
            st.write(str(result.get("dosage", "No dosage details available.")))
        with rec_col2:
            st.markdown("### 🛡️ Long-Term Prevention")
            st.write(str(result.get("prevention", "No preventative steps available.")))
        st.markdown("""</div>""", unsafe_allow_html=True)

    with tab_ai:
        st.markdown("""<div class="organic-card" style="padding: 30px;">""", unsafe_allow_html=True)
        ai_text = result.get("ai_analysis")
        if ai_text:
            st.markdown(ai_text)
        else:
            st.info("AI Agronomist is loading or was not queried. Please make sure GEMINI_API_KEY is configured in your .env file to unlock rich advisory reports.")
        st.markdown("""</div>""", unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit application."""
    _page_config()
    _inject_custom_css()
    _render_header()

    image_bytes, filename = _get_image_bytes()
    if image_bytes is None or filename is None:
        st.info("Please upload or capture a crop root or soil core image to begin diagnostics.")
        return

    if st.button("Run Diagnostic Scan", type="primary"):
        with st.spinner("Analyzing root & soil parameters..."):
            try:
                result = _predict(image_bytes=image_bytes, filename=filename)
            except requests.RequestException:
                st.error(
                    "Could not reach the AgriScan API. Ensure the backend is active."
                )
                return
            except RuntimeError as exc:
                error_msg = str(exc)
                st.error(f"❌ {error_msg}")
                st.info(
                    "💡 **Tips for best results:**\n"
                    "- Upload a **clear, close-up photo** of an affected root system or soil sample\n"
                    "- Ensure the root/soil is **well-lit** and **in focus**\n"
                    "- Avoid shadows or blurred images\n"
                    "- Make sure you're uploading a **crop root or soil sample, not a leaf or person**"
                )
                return

        _render_results(result=result, image_bytes=image_bytes)


if __name__ == "__main__":
    main()
