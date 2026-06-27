# 🌿 AgriScan AI — Crop Disease & Pest Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**An AI-powered, software-only crop diagnostic tool that gives any farmer instant pest and disease identification from a single leaf photo — along with a clear, actionable treatment recommendation.**

[Live Demo](#) • [Documentation](#getting-started) • [Report Bug](#) • [Request Feature](#)

</div>

---

## � Table of Contents

- [✨ About the Project](#-about-the-project)
- [🎯 Problem Statement](#-problem-statement)
- [💡 Solution](#-solution)
- [⭐ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🔄 How It Works](#-how-it-works)
- [📊 Dataset](#-dataset)
- [🤖 Model Details](#-model-details)
- [📸 Screenshots](#-screenshots)
- [👥 Team](#-team)
- [🗓️ Future Roadmap](#️-future-roadmap)
- [📄 License](#-license)

---

## ✨ About the Project

**AgriScan AI** is a submission for [Biothon 2026](https://marwadiuniversity.ac.in) (Agriculture Domain), organized by Marwadi University, sponsored by the Royal Society of Biology and IEEE EMBS.

### The Problem We're Solving

🌾 **Indian agriculture at risk:** Farmers lose an estimated **15–25% of crop yield annually** to pests and diseases that go undetected until visible damage has already spread.

💔 **Limited access to expertise:** Most small and marginal farmers lack access to agronomists or diagnostic labs and rely on guesswork—leading to:
- Excessive pesticide use
- Wrongly-timed treatments
- Avoidable economic losses

### Our Solution

✅ **AgriScan AI** — A smartphone-accessible, **zero-hardware web tool** that:
1. Identifies crop disease or pest from a leaf photo
2. Instantly returns a treatment plan
3. Requires **no installation**, **no hardware**, works on low-bandwidth networks

**Built for India's farming community.** Accessible on any smartphone browser.

---

## 🎯 Problem Statement

- Expert agronomist visits are slow, expensive, and unavailable in rural areas
- Existing plant disease apps only classify the disease — they don't recommend treatment
- Hardware-based precision agriculture solutions are cost-prohibitive for small farmers
- No affordable, end-to-end diagnosis-to-treatment tool exists at scale for India's farming community

---

## 💡 Solution

AgriScan AI provides a simple yet powerful workflow for crop disease diagnosis:

### How a Farmer Uses AgriScan AI

1. 📱 Opens the app on any smartphone browser (no installation required)
2. 📸 Uploads or captures a photo of an affected crop leaf
3. ⚡ Receives instant disease/pest classification with confidence score
4. 💡 Gets a clear treatment recommendation:
   - Specific remedy name
   - Dosage details
   - Preventive steps for future

### Why It Works

- **Accessible** — Works on any smartphone, even with poor connectivity
- **Affordable** — No hardware costs, no expert consultation fees
- **Actionable** — Not just diagnosis, but clear next steps for treatment
- **Inclusive** — Designed specifically for small and marginal farmers

---

## ⭐ Features

### 🎯 Core Capabilities

- **⚡ Instant Detection**  
  YOLOv8-based computer vision model classifies pest/disease within seconds

- 💊 **Treatment Advisory**  
  Every diagnosis is mapped to a structured remedy guide (organic + chemical options)

- 📊 **Confidence Scoring**  
  Model output includes confidence score so farmers know how reliable the result is

- 📱 **Browser-Based UI**  
  Streamlit frontend accessible from any smartphone browser, no app installation needed

- 🚀 **Lightweight Backend**  
  FastAPI REST API designed for low-latency inference on standard hardware

- 🔧 **Extensible Architecture**  
  Add new crops and disease classes by retraining on additional labeled data

### 🌟 Key Advantages

✅ Zero hardware required  
✅ Works offline (PWA version planned)  
✅ Supports low-bandwidth networks  
✅ Treatment recommendations built-in  
✅ Confidence scoring for reliability  
✅ Easy to retrain with new data  

---

## 🛠️ Tech Stack

| **Layer** | **Technology** | **Purpose** |
|---|---|---|
| **Frontend** | Streamlit | User interface & image upload |
| **Backend API** | FastAPI (Python) | High-performance REST API |
| **Machine Learning** | YOLOv8 (Ultralytics) | Object detection & classification |
| **Computer Vision** | OpenCV, PyTorch | Image processing & inference |
| **Data Labeling** | Roboflow | Annotation & dataset management |
| **Training** | Google Colab (GPU) | Model training environment |
| **Knowledge Base** | SQLite + JSON | Disease-to-treatment mapping |
| **Version Control** | Git / GitHub | Code management |
| **Deployment** | Streamlit Cloud / Render | Cloud hosting |

---

## 📁 Project Structure

```
agriscan-ai/
│
├── app/                          # 🔧 Backend application
│   ├── main.py                   # FastAPI entry point & routes
│   ├── inference.py              # YOLOv8 model loading & prediction
│   ├── treatment.py              # Treatment recommendation lookup
│   └── models/
│       └── best.pt               # Trained YOLOv8 weights ⚠️ (not in repo)
│
├── frontend/                     # 🎨 User interface
│   └── streamlit_app.py          # Streamlit web application
│
├── data/                         # 📊 Knowledge & training data
│   ├── knowledge_base.json       # Disease → treatment mapping
│   └── classes.txt               # Model class labels
│
├── training/                     # 🤖 Model training
│   └── train.ipynb               # Google Colab training notebook
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # Project documentation
```

### 📝 Key Files Description

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI server with `/predict` endpoint |
| `app/inference.py` | Loads & runs YOLOv8 model for disease detection |
| `app/treatment.py` | Maps diseases to treatment recommendations |
| `frontend/streamlit_app.py` | Web UI for image upload & result display |
| `data/knowledge_base.json` | Database mapping diseases to remedies |
| `training/train.ipynb` | Notebook for YOLOv8 model training on Colab |

> **⚠️ Note:** The trained model weights (`best.pt`) are not committed to the repository due to file size. Download instructions are in [Getting Started](#-getting-started).

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **pip** (Python package manager)
- **Git** (version control)

### Installation Steps

#### Step 1️⃣ — Clone the Repository

```bash
git clone https://github.com/bhoomika0910/agriscan-ai.git
cd agriscan-ai
```

#### Step 2️⃣ — Create Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3️⃣ — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4️⃣ — Configure Environment Variables

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Update `.env` as needed for:

- API host and port
- backend model and knowledge base paths
- frontend API URL
- allowed CORS origins

#### Step 5️⃣ — Download Model Weights

Download the trained `best.pt` file and place it in `app/models/`:

- **Option A:** Download from [GitHub Releases](https://github.com/bhoomika0910/agriscan-ai/releases)
- **Option B:** Train your own using `training/train.ipynb`

#### Step 6️⃣ — Run FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ API will be available at `http://localhost:8000`  
📚 API docs: `http://localhost:8000/docs`

#### Step 7️⃣ — Run Streamlit Frontend

Open a new terminal and run:

```bash
streamlit run frontend/streamlit_app.py
```

✅ Frontend will open at `http://localhost:8501`

---

## 🔄 How It Works

### Complete Workflow

```
┌─────────────────────────────────────┐
│  👨‍🌾 Farmer captures leaf photo     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  📱 Streamlit Web UI (Frontend)      │
│   - Image upload/capture            │
│   - User-friendly interface         │
└──────────┬──────────────────────────┘
           │
           ├─→ HTTP POST /predict
           │
           ▼
┌─────────────────────────────────────┐
│  ⚡ FastAPI Backend (app/main.py)    │
└──────────┬──────────────────────────┘
           │
           ├─→ inference.py            │
           │   ├─ Load YOLOv8 model    │
           │   └─ Run prediction       │
           │                           │
           ├─→ treatment.py            │
           │   ├─ Look up disease      │
           │   └─ Fetch remedy details │
           │
           ▼
┌─────────────────────────────────────┐
│  📊 Response Object                 │
│  {                                  │
│    "disease": "Early Blight",       │
│    "confidence": 0.94,              │
│    "remedy": "Copper fungicide",    │
│    "dosage": "2.5L per acre",       │
│    "prevention": "Rotate crops"     │
│  }                                  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  ✅ Farmer receives treatment plan  │
└─────────────────────────────────────┘
```

### Data Flow Explanation

1. **Image Capture** → User uploads leaf image via Streamlit UI
2. **API Request** → Frontend sends image to FastAPI backend
3. **Model Inference** → YOLOv8 classifies the disease and returns confidence score
4. **Treatment Lookup** → Backend queries knowledge_base.json for remedy details
5. **Response Delivery** → Results displayed with clear, actionable recommendations

---

## 📊 Dataset

### Primary Source: PlantVillage Dataset

We leverage the **PlantVillage Dataset** as our primary training source for crop disease classification.

| Metric | Value |
|--------|-------|
| **Total Images** | 54,000+ labeled images |
| **Crop Types** | Tomato, Potato, Maize, Pepper, and more |
| **Disease Classes** | 38 unique crop disease classes |
| **Labeling Tool** | [Roboflow](https://roboflow.com) |
| **Data Source** | [Kaggle - PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) |

### Using Custom Datasets

To train with your own dataset:

1. Collect and label your leaf images
2. Export dataset from **Roboflow** in **YOLOv8 format**
3. Place dataset in `data/` directory
4. Update dataset path in `training/train.ipynb`
5. Run training notebook on Google Colab

See [classes.txt](data/classes.txt) for current model class labels.

---

## 🤖 Model Details

### Model Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | YOLOv8 Nano (Lightweight) |
| **Input Size** | 640 × 640 pixels |
| **Training Epochs** | 50 |
| **Optimizer** | AdamW |
| **Batch Size** | 16 |
| **Training Hardware** | Google Colab (NVIDIA T4 GPU) |
| **Inference Speed** | ~50-100ms per image |
| **Evaluation Metric** | mAP@50 (Mean Average Precision) |
| **Framework** | PyTorch + Ultralytics YOLOv8 |

### Why YOLOv8 Nano?

✅ **Lightweight** — Ideal for mobile & low-bandwidth rural networks  
✅ **Fast** — Real-time inference (50-100ms per image)  
✅ **Accurate** — Maintains high mAP while being lightweight  
✅ **Portable** — Easy to deploy and export  

### Training & Export

The full training pipeline is available in `training/train.ipynb`:

- ✏️ Data loading & augmentation
- 🎓 Model training on Colab GPU
- 📊 Evaluation & metrics
- 💾 Export to `.pt` format for inference

See [training/train.ipynb](training/train.ipynb) for complete implementation.

---

## 📸 Screenshots

> Coming soon — will be added post-prototype build.

---

## 👥 Team

**Team Dumbos** — Biothon 2026, Agriculture Domain  
Marwadi University, Sponsored by Royal Society of Biology & IEEE EMBS

| Name | Role | College | Contact |
|------|------|--------|---------|
| **Bhoomika Agarwal** | Team Lead, AI/ML & Backend | GLA University, Mathura | [GitHub](https://github.com/bhoomika0910) |
| **Amrita Singh** | Frontend & Data Pipeline | GLA University, Mathura | TBA |

---

## 🗓️ Future Roadmap

### Phase 1️⃣ — Core Features (Current)
- [x] YOLOv8 disease classification model
- [x] FastAPI backend with inference
- [x] Streamlit web UI
- [x] Knowledge base for treatments

### Phase 2️⃣ — Expansion & Localization (Q3 2026)
- [ ] Expand model to cover wheat, rice, sugarcane
- [ ] Add Hindi & regional language support
- [ ] Offline-capable Progressive Web App (PWA)
- [ ] Mobile app for Android/iOS

### Phase 3️⃣ — Community & Scaling (Q4 2026)
- [ ] Crowdsourced image upload system
- [ ] Continuous model improvement pipeline
- [ ] Community feedback mechanism
- [ ] Field performance tracking

### Phase 4️⃣ — Government & Enterprise (2027)
- [ ] B2G API licensing for state agriculture departments
- [ ] Partnership with KVKs (Krishi Vigyan Kendras)
- [ ] Integration with agri-NGOs
- [ ] Large-scale field deployment pilot

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

You are free to:
- ✅ Use this project for commercial purposes
- ✅ Modify and distribute the code
- ✅ Use privately
- ✅ Sublicense

**With the condition that you include a copy of the license and copyright notice.**

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, or code improvements, your help makes AgriScan AI better for farmers everywhere.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas We Need Help With

- 🖼️ UI/UX improvements
- 🤖 Model accuracy enhancement
- 🌍 Localization (Hindi, regional languages)
- 📊 Dataset expansion
- 🐛 Bug fixes
- 📚 Documentation

Please ensure your code follows our style guidelines and includes appropriate tests.

---

## 📞 Support & Contact

### Get Help

- 📖 **Documentation** — See [Getting Started](#-getting-started)
- 🐛 **Report Issues** — [GitHub Issues](https://github.com/bhoomika0910/agriscan-ai/issues)
- 💬 **Discussions** — [GitHub Discussions](https://github.com/bhoomika0910/agriscan-ai/discussions)
- 📧 **Email** — [Project Lead](mailto:bhoomika0910@gmail.com)

### Quick Links

- [Project Repository](https://github.com/bhoomika0910/agriscan-ai)
- [Dataset (PlantVillage)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [FastAPI Docs](https://fastapi.tiangolo.com)

---

## 📊 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Model | ✅ Complete | YOLOv8 trained & working |
| Backend API | ✅ Complete | FastAPI fully functional |
| Frontend UI | ✅ Complete | Streamlit interface ready |
| Deployment | 🔄 In Progress | Cloud deployment setup |
| Localization | 📋 Planned | Hindi + regional languages |

---

<div align="center">

### 💚 Built with ❤️ for Indian Farmers

**Biothon 2026 · Agriculture Domain · Team Dumbos**

*Empowering farmers with AI-driven crop disease diagnosis*

[⬆ back to top](#-agriscan-ai--crop-disease--pest-detector)

</div>
