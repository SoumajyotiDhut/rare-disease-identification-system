from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import numpy as np
from datetime import datetime
from typing import Optional
import os

app = FastAPI(title="AI DOC — Rare Disease API")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ──
# Models loaded only on first request, not on startup
_tfidf      = None
_lr_model   = None
_le         = None
_history    = []

def load_models():
    """Load models once, cache in memory"""
    global _tfidf, _lr_model, _le

    if _tfidf is not None:
        return  # already loaded

    print("Loading models...")
    try:
        base = os.path.dirname(__file__)

        with open(os.path.join(base, "models/tfidf_vectorizer.pkl"), "rb") as f:
            _tfidf = pickle.load(f)

        with open(os.path.join(base, "models/lr_symptoms_model.pkl"), "rb") as f:
            _lr_model = pickle.load(f)

        with open(os.path.join(base, "models/label_encoder_symptoms.pkl"), "rb") as f:
            _le = pickle.load(f)

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


def get_confidence(prob: float) -> str:
    if prob >= 0.5:  return "High"
    if prob >= 0.2:  return "Medium"
    return "Low"


# ── Routes ──

@app.get("/")
def root():
    return {"status": "ok", "service": "AI DOC Rare Disease API"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict/text")
async def predict_text(
    symptoms: str = Form(...),
    top_k: int    = Form(5),
):
    load_models()  # lazy load

    # Preprocess
    text = " [SEP] ".join(
        [s.strip().lower() for s in symptoms.replace(",", "\n").split("\n") if s.strip()]
    )

    vec   = _tfidf.transform([text])
    proba = _lr_model.predict_proba(vec)[0]
    top_k = min(top_k, len(proba))

    top_idx   = np.argsort(proba)[-top_k:][::-1]
    predictions = []

    for rank, idx in enumerate(top_idx, 1):
        orpha    = _le.inverse_transform([idx])[0]
        prob_pct = round(float(proba[idx]) * 100, 1)
        predictions.append({
            "rank"       : rank,
            "disease"    : f"ORPHA:{orpha}",
            "orpha_code" : str(orpha),
            "probability": prob_pct,
            "confidence" : get_confidence(proba[idx]),
        })

    # Save to history
    _history.append({
        "timestamp"  : datetime.utcnow().isoformat(),
        "symptoms"   : symptoms,
        "predictions": predictions,
        "mode"       : "symptoms_only",
    })

    return {"predictions": predictions, "mode": "symptoms_only"}


@app.post("/predict")
async def predict_multimodal(
    symptoms: str                    = Form(...),
    top_k: int                       = Form(5),
    image: Optional[UploadFile]      = File(None),
):
    """
    Multimodal endpoint — currently uses symptoms model only.
    Image model will be added after training completes.
    """
    return await predict_text(symptoms=symptoms, top_k=top_k)


@app.get("/analytics")
def analytics():
    total = len(_history)
    high  = sum(
        1 for h in _history
        if (h.get("predictions") or [{}])[0].get("confidence") == "High"
    )
    return {
        "total_predictions" : total,
        "high_confidence"   : high,
        "diseases_covered"  : 49,
        "models_active"     : 1,
    }


@app.get("/history")
def history():
    return {"history": _history[-50:]}  # last 50 only