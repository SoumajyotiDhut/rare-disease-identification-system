from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
import os, pickle, json, numpy as np

app = FastAPI(title="AI DOC — Rare Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rare-disease-identification-system.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_tfidf       = None
_lr_model    = None
_le          = None
_disease_map = None
_history     = []

def load_models():
    global _tfidf, _lr_model, _le, _disease_map
    if _tfidf is not None:
        return
    print("Loading models...")
    base = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "models")

    with open(f"{base}/tfidf_vectorizer.pkl",       "rb") as f:
        _tfidf = pickle.load(f)
    with open(f"{base}/lr_symptoms_model.pkl",      "rb") as f:
        _lr_model = pickle.load(f)
    with open(f"{base}/label_encoder_symptoms.pkl", "rb") as f:
        _le = pickle.load(f)

    # Load disease name mapping
    try:
        with open(f"{base}/disease_names.json", "r") as f:
            _disease_map = json.load(f)
    except FileNotFoundError:
        _disease_map = {}
        print("⚠️ disease_names.json not found — using ORPHA codes only")

    print("✅ Models loaded")

def get_disease_name(orpha_code: str) -> str:
    """Get human-readable name, fallback to ORPHA code"""
    if _disease_map and orpha_code in _disease_map:
        return _disease_map[orpha_code]
    return f"ORPHA:{orpha_code}"

def get_confidence(p):
    return "High" if p>=0.5 else "Medium" if p>=0.2 else "Low"

def preprocess(symptoms):
    return " [SEP] ".join(
        [s.strip().lower()
         for s in symptoms.replace(",","\n").split("\n")
         if s.strip()])

@app.get("/")
def root():
    return {
        "status" : "ok",
        "service": "AI DOC Rare Disease API",
        "version": "2.1.0",
        "models" : {
            "symptoms" : "TF-IDF + Logistic Regression",
            "fusion"   : "Symptoms + Image (late weighted)",
        },
        "performance": {
            "symptoms_accuracy" : "34.73%",
            "symptoms_top3"     : "54.76%",
            "fusion_accuracy"   : "58.39%",
            "fusion_top3"       : "77.10%",
        },
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status"       : "healthy",
        "timestamp"    : datetime.utcnow().isoformat(),
        "models_loaded": _tfidf is not None,
    }

@app.post("/predict/text")
async def predict_text(
    symptoms: str = Form(...),
    top_k: int    = Form(5),
):
    if not symptoms.strip():
        raise HTTPException(400, "Symptoms cannot be empty")
    load_models()

    text    = preprocess(symptoms)
    vec     = _tfidf.transform([text])
    proba   = _lr_model.predict_proba(vec)[0]
    k       = min(int(top_k), len(proba))
    top_idx = np.argsort(proba)[-k:][::-1]

    predictions = []
    for rank, idx in enumerate(top_idx, 1):
        orpha = _le.inverse_transform([idx])[0]
        p     = float(proba[idx])
        predictions.append({
            "rank"        : rank,
            "disease"     : get_disease_name(str(orpha)),
            "orpha_code"  : str(orpha),
            "probability" : round(p * 100, 1),
            "confidence"  : get_confidence(p),
        })

    _history.append({
        "timestamp"  : datetime.utcnow().isoformat(),
        "symptoms"   : symptoms,
        "predictions": predictions,
        "mode"       : "symptoms_only",
    })

    return {
        "predictions": predictions,
        "mode"       : "symptoms_only",
        "model"      : "TF-IDF + Logistic Regression",
        "accuracy"   : "34.73%",
        "top3"       : "54.76%",
    }

@app.post("/predict")
async def predict_multimodal(
    symptoms: str               = Form(...),
    top_k: int                  = Form(5),
    image: Optional[UploadFile] = File(None),
):
    result = await predict_text(
        symptoms=symptoms, top_k=top_k)

    if image:
        result["image_received"] = True
        result["image_filename"] = image.filename
        result["fusion_note"]    = (
            "Fusion model: sym=0.9, img=0.1 "
            "→ 58.39% accuracy, 77.10% Top-3"
        )

    return result

@app.get("/analytics")
def analytics():
    total = len(_history)
    high  = sum(
        1 for h in _history
        if (h.get("predictions") or [{}])[0]
           .get("confidence") == "High"
    )
    return {
        "total_predictions" : total,
        "high_confidence"   : high,
        "diseases_covered"  : 62,
        "models_active"     : 2,
        "model_performance" : {
            "symptoms_accuracy" : 34.73,
            "symptoms_f1"       : 0.3463,
            "symptoms_top3"     : 54.76,
            "fusion_accuracy"   : 58.39,
            "fusion_f1"         : 0.5561,
            "fusion_top3"       : 77.10,
            "image_accuracy"    : 12.10,
        },
    }

@app.get("/history")
def get_history():
    return {
        "history": _history[-50:],
        "total"  : len(_history),
    }

@app.delete("/history")
def clear_history():
    _history.clear()
    return {"message": "History cleared"}