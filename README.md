# 🧬 AI DOC — Rare Disease Identification System

> An AI-powered multimodal diagnostic assistant that identifies rare diseases from patient symptoms and biomedical images, built on the ZebraMap dataset (36,487 real patient case reports across 1,374 rare diseases).

[![Live Demo](https://img.shields.io/badge/Frontend-Live%20on%20Vercel-00D4C8?style=flat-square)](https://rare-disease-identification-system.vercel.app)
[![API](https://img.shields.io/badge/Backend-Hugging%20Face%20Spaces-FFD21E?style=flat-square)](https://soumadhut-ai-doc-rare-disease-api.hf.space)
[![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue?style=flat-square)](https://zenodo.org/records/17623607)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Models & Results](#-models--results)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Local Setup](#-local-setup)
- [API Reference](#-api-reference)
- [Experiments](#-experiments)
- [Roadmap](#-roadmap)
- [Acknowledgments](#-acknowledgments)

---

## 🩺 Overview

Rare diseases affect **1 in 17 people worldwide**, yet patients wait an average of **5–7 years** for an accurate diagnosis due to limited clinician familiarity with thousands of rare conditions. **AI DOC** addresses this by combining two diagnostic signals doctors already use — **patient-reported symptoms** and **biomedical imaging** — into a single AI-powered differential diagnosis tool.

The system uses **three independently trained models**:

| Model | Input | Purpose |
|---|---|---|
| 🧾 **Symptoms Model** | Free-text symptom list | TF-IDF + Logistic Regression classifier |
| 🩻 **Image Model** | Medical scan (MRI/CT/Histopathology/etc.) | EfficientNet-B4 fine-tuned classifier |
| 🔗 **Fusion Model** | Both combined | Late-weighted fusion of both models' outputs |

Fusing both modalities improves Top-1 accuracy from **34.73% → 58.39%** and Top-3 accuracy from **54.76% → 77.10%** over symptoms alone — proving multimodal diagnosis significantly outperforms either signal in isolation.

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| 🌐 **Frontend** | https://rare-disease-identification-system.vercel.app |
| ⚙️ **Backend API** | https://soumadhut-ai-doc-rare-disease-api.hf.space |
| 📚 **API Docs (Swagger)** | https://soumadhut-ai-doc-rare-disease-api.hf.space/docs |

**Try it:** Go to `/predict`, enter `night blindness, progressive visual field loss, bone spicule pigmentation` → returns **Retinitis Pigmentosa at ~92% confidence**.

---

## 🔬 How It Works

```
┌─────────────────────┐       ┌─────────────────────┐
│   Patient Symptoms   │       │   Medical Image      │
│  "night blindness,    │       │  (MRI/CT/Dermoscopy/ │
│   visual field loss"  │       │   Histopathology)     │
└──────────┬──────────┘       └──────────┬──────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐             ┌───────────────────┐
   │  TF-IDF Vector │             │  EfficientNet-B4    │
   │  + Logistic     │             │  (pretrained on      │
   │  Regression     │             │  ImageNet, fine-     │
   │                 │             │  tuned on 35K scans) │
   └───────┬────────┘             └─────────┬──────────┘
           │  62-dim probability             │  62-dim probability
           ▼                                  ▼
           └───────────────┬──────────────────┘
                            ▼
              ┌──────────────────────────┐
              │     Late Fusion Layer      │
              │ score = 0.9×sym + 0.1×img  │
              └─────────────┬────────────┘
                            ▼
              ┌──────────────────────────┐
              │  Ranked Disease Predictions │
              │  with confidence scores      │
              └──────────────────────────┘
```

If only symptoms are provided, the system falls back gracefully to the symptoms-only model. If an image is also uploaded, the fusion pathway activates automatically.

---

## 📊 Dataset

**[ZebraMap](https://zenodo.org/records/17623607)** — A Multimodal Rare Disease Knowledge Map (Zenodo, CC BY 4.0), built by linking Orphanet rare diseases to PubMed open-access case reports via an LLM-based extraction pipeline.

| Metric | Value |
|---|---|
| Total patient case reports | 36,487 |
| Unique rare diseases (ORPHA codes) | 1,374 |
| Cases with verified images + symptoms | 23,770 |
| Total biomedical images | 94,384 |
| Avg. symptoms per case | 8.5 |
| Avg. images per case | 2.5 |

**Disease tiers** (used to simulate real-world data scarcity):

| Tier | Cases per disease | Diseases | Purpose |
|---|---|---|---|
| **A** | ≥ 100 | 62 | Primary training set |
| **B** | 30–99 | 126 | GAN augmentation target |
| **C** | < 30 | 1,069 | Ultra-rare disease validation |

---

## 🏆 Models & Results

| Experiment | Model | Accuracy | Macro F1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| Exp 1 | Symptoms — 5% data (scarcity sim.) | 24.83% | 0.2240 | 40.27% | — |
| Exp 3 | Symptoms — 100% data | **34.73%** | 0.3463 | 54.76% | 62.67% |
| Exp 3 | Image only (EfficientNet-B4) | 12.10% | 0.1098 | 25.00% | 32.10% |
| **Fusion** | **Symptoms + Image (late weighted)** | **58.39%** | **0.5561** | **77.10%** | **83.87%** |

**Key findings:**
- More training data significantly improves performance (+9.23% accuracy, Exp1→Exp3) — directly demonstrating the rare disease data scarcity problem.
- Fusion of both modalities outperforms either alone by a wide margin (+23.66% over symptoms-only), proving multimodal diagnosis is the right approach for rare disease identification.
- Optimal fusion weighting found via grid search: **w_symptoms = 0.9, w_image = 0.1** (the symptom signal carries more discriminative information than images at current image-model accuracy).

---

## 📁 Project Structure

```
rare-disease-identification-system/
│
├── backend/                          # FastAPI backend (legacy / Render deploy)
│   ├── app.py                        # Main API — symptoms + fusion endpoints
│   ├── model_loader.py               # Heavy PyTorch model loading utilities
│   ├── predict_fusion.py             # Fusion prediction logic
│   ├── database.py                   # SQLite history persistence
│   ├── resave_model.py               # Re-saves models for sklearn version compat
│   ├── resave_label_encoder.py
│   ├── resave_nlp_v2.py
│   ├── upload_*.py                   # One-off scripts to push models to HF Hub
│   ├── Dockerfile
│   ├── Procfile
│   ├── requirements.txt
│   └── models/                       # Local model artifacts (gitignored — large files)
│       ├── tfidf_vectorizer.pkl
│       ├── lr_symptoms_model.pkl
│       ├── label_encoder_symptoms.pkl
│       ├── label_encoder.pkl
│       ├── label_encoder_v2.pkl
│       ├── image_model_best.pt
│       ├── fusion_model.pt / fusion_model_clean.pt
│       ├── improved_nlp_v2.pt / improved_nlp_v2_clean.pt
│       ├── cnn_v2_full.pt
│       └── disease_names.json        # ORPHA code → human-readable disease name
│
├── hf-space/                         # 🚀 PRODUCTION backend — Hugging Face Spaces
│   ├── app.py                        # FastAPI app (symptoms + image fusion, live)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md                     # HF Space config (SDK, hardware)
│
├── rare-disease-models/              # Model artifact mirror pushed to HF Hub (git-lfs)
│   ├── tfidf_vectorizer.pkl
│   ├── lr_symptoms_model.pkl
│   ├── image_model_best.pt
│   ├── fusion_model.pt
│   ├── improved_nlp_v2.pt
│   ├── cnn_v2_full.pt
│   ├── label_encoder*.pkl
│   └── disease_names.json
│
├── frontend/                         # React + Vite frontend — deployed on Vercel
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx              # Landing page with live stats
│   │   │   ├── Predict.jsx           # Main diagnosis interface
│   │   │   ├── Dashboard.jsx         # Model performance analytics
│   │   │   └── History.jsx           # Past prediction sessions
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── Loader.jsx
│   │   │   ├── PageTransition.jsx
│   │   │   └── PredictionCard.jsx
│   │   ├── context/
│   │   │   ├── ThemeContext.jsx      # Dark/light mode
│   │   │   └── ToastContext.jsx      # Notification toasts
│   │   ├── services/
│   │   │   └── Api.js                # Backend API client
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── theme.js
│   │   └── main.jsx
│   ├── public/
│   ├── dist/                         # Production build output
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
├── notebooks/                        # All model development notebooks (chronological)
│   ├── colab/
│   │   ├── Day_1.ipynb                          # Project setup
│   │   ├── Day_2_EDA_NLP_Baseline.ipynb          # Data exploration
│   │   ├── Day_3_NLP_Baseline.ipynb              # Symptoms model v1
│   │   ├── Day_4_CNN_Baseline.ipynb              # Image model v1
│   │   ├── Day_4_image_model.ipynb               # Image model (final, Tier A)
│   │   ├── Day_5_Full_Dataset_*.ipynb            # Full dataset preprocessing
│   │   ├── Day_7_ZebraMap_*.ipynb                # ZebraMap integration
│   │   ├── Day_8_Results_Dashboard.ipynb
│   │   ├── Day_9_GAN_Setup.ipynb                 # GAN architecture (planned)
│   │   ├── Day_12_ZebraMap_*.ipynb
│   │   ├── Day_16_17_Final_*.ipynb
│   │   └── Day_18_Fusion_model.ipynb             # ✅ Final fusion model & results
│   └── Kaggle/
│       ├── Day_5_Cnn_Upperbound.ipynb
│       ├── Day_6_Ham10000_*.ipynb
│       ├── Day_7_zebramap_*.ipynb
│       ├── Day_10_Ham10000_*.ipynb
│       ├── Day_11_zebramap_*.ipynb
│       └── Day_13_14_15_*.ipynb                  # Symptoms model training (GPU)
│
├── url                                # Saved reference URLs (deploy endpoints, dataset link)
├── .python-version
├── .gitignore
├── package.json / package-lock.json   # Root-level scripts (if any)
└── README.md                          # You are here
```

---

## 🛠️ Tech Stack

**Backend / ML**
- Python · FastAPI · Uvicorn
- scikit-learn (TF-IDF, Logistic Regression)
- PyTorch · timm (EfficientNet-B4)
- Hugging Face Hub (model hosting via git-lfs)
- Docker (Hugging Face Spaces deployment)

**Frontend**
- React 18 · Vite
- Tailwind CSS
- React Router
- Recharts (analytics charts)
- Lucide React / React Icons

**Data & Training**
- Google Colab + Google Drive (exploration, preprocessing)
- Kaggle (GPU model training)
- ZebraMap dataset (Zenodo, CC BY 4.0)

**Deployment**
- Frontend → **Vercel**
- Backend → **Hugging Face Spaces** (Docker SDK, CPU, 16GB RAM)

---

## ⚙️ Local Setup

### Backend (Hugging Face Space version)

```bash
cd hf-space
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

API will be live at `http://localhost:7860`. Visit `/docs` for interactive Swagger UI.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App will be live at `http://localhost:5173`.

> Update `frontend/src/services/Api.js` → `BASE_URL` to point to `http://localhost:7860` for local backend testing.

---

## 📡 API Reference

### `POST /predict/text`
Symptom-only prediction.

```bash
curl -X POST "https://soumadhut-ai-doc-rare-disease-api.hf.space/predict/text" \
  -F "symptoms=night blindness, progressive visual field loss" \
  -F "top_k=5"
```

```json
{
  "predictions": [
    { "rank": 1, "disease": "Retinitis pigmentosa", "orpha_code": "791", "probability": 92.0, "confidence": "High" }
  ],
  "mode": "symptoms_only",
  "model": "TF-IDF + Logistic Regression",
  "accuracy": "34.73%",
  "top3": "54.76%"
}
```

### `POST /predict`
Multimodal prediction — symptoms + optional image. Falls back to symptoms-only if no image is sent.

```bash
curl -X POST "https://soumadhut-ai-doc-rare-disease-api.hf.space/predict" \
  -F "symptoms=fever, fatigue, joint pain" \
  -F "image=@scan.jpg" \
  -F "top_k=5"
```

### `GET /analytics`
Returns platform-wide prediction stats and model performance metrics.

### `GET /history`
Returns the last 50 prediction sessions.

### `GET /health`
Service health check (used for uptime monitoring).

Full interactive documentation: **[/docs](https://soumadhut-ai-doc-rare-disease-api.hf.space/docs)**

---

## 🧪 Experiments

| # | Name | Tier | Data | GAN | Status |
|---|---|---|---|---|---|
| 1 | Scarcity baseline | A | 5% (514 samples) | ❌ | ✅ Done |
| 2 | GAN augmented | A | 5% + synthetic | ✅ | ⏳ Planned |
| 3 | Upper bound | A | 100% (8,568 / 10,711 samples) | ❌ | ✅ Done |
| 4 | Real rare-disease baseline | C | Full (3,759 samples) | ❌ | ⏳ Planned |
| 5 | Real rare-disease + GAN | C | Full + synthetic | ✅ | ⏳ Planned |
| — | Fusion (symptoms + image) | A | 3,100 images, 62 diseases | ❌ | ✅ Done |

---

## 🗺️ Roadmap

- [x] Dataset acquisition & multimodal alignment (ZebraMap)
- [x] Symptoms model (TF-IDF + Logistic Regression)
- [x] Image model (EfficientNet-B4)
- [x] Late-fusion multimodal model
- [x] FastAPI backend with lazy model loading
- [x] React frontend with live predictions, dashboard, history
- [x] Production deployment (Vercel + Hugging Face Spaces)
- [ ] GAN-based data augmentation for Tier B/C diseases (Exp 2, 5)
- [ ] Ultra-rare disease baseline on Tier C (Exp 4)
- [ ] Persistent history storage (currently in-memory)
- [ ] Cross-attention fusion (replace late-weighted fusion)

---

## 🙏 Acknowledgments

- **[ZebraMap](https://zenodo.org/records/17623607)** dataset authors — for building and openly licensing this multimodal rare disease resource (CC BY 4.0).
- **[Orphanet](https://www.orpha.net/)** — rare disease nomenclature (ORPHA codes) and reference data.
- **PubMed Central** — source of the underlying open-access case reports aggregated by ZebraMap.

---

## 📄 License

This project uses the ZebraMap dataset under **CC BY 4.0**. Application code is provided for academic/research purposes.

---

<p align="center">Built as a final-year project exploring multimodal AI for rare disease diagnosis.</p>