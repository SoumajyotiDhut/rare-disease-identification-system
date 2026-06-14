from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from datetime import datetime
from typing import Optional
import os
import io

app = FastAPI(title="AI DOC — Rare Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model cache ──
_tfidf    = None
_lr_model = None
_le       = None
_history  = []

def load_models():
    global _tfidf, _lr_model, _le
    if _tfidf is not None:
        return

    print("Loading models...")
    base        = os.path.dirname(os.path.abspath(__file__))
    models_dir  = os.path.join(base, "models")
    hf_repo     = os.environ.get("HF_REPO", "soumajyotidhut/rare-disease-models")
    hf_token    = os.environ.get("HF_TOKEN", None)

    # ── Try local files first ──
    local_files = {
        "tfidf"   : os.path.join(models_dir, "tfidf_vectorizer.pkl"),
        "lr"      : os.path.join(models_dir, "lr_symptoms_model.pkl"),
        "le"      : os.path.join(models_dir, "label_encoder_symptoms.pkl"),
    }

    all_local = all(os.path.exists(p) for p in local_files.values())

    if all_local:
        print("Loading from local models/ folder...")
        with open(local_files["tfidf"], "rb") as f:
            _tfidf = pickle.load(f)
        with open(local_files["lr"], "rb") as f:
            _lr_model = pickle.load(f)
        with open(local_files["le"], "rb") as f:
            _le = pickle.load(f)
        print("✅ Models loaded from local files")

    else:
        # ── Fallback: download from Hugging Face ──
        print(f"Local models not found. Downloading from HF: {hf_repo}")
        try:
            from huggingface_hub import hf_hub_download

            def load_pkl(filename):
                path = hf_hub_download(
                    repo_id  = hf_repo,
                    filename = filename,
                    token    = hf_token,
                )
                with open(path, "rb") as f:
                    return pickle.load(f)

            _tfidf    = load_pkl("tfidf_vectorizer.pkl")
            _lr_model = load_pkl("lr_symptoms_model.pkl")
            _le       = load_pkl("label_encoder_symptoms.pkl")
            print("✅ Models loaded from Hugging Face")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise HTTPException(
                status_code = 503,
                detail      = f"Models unavailable: {str(e)}"
            )


def get_confidence(prob: float) -> str:
    if prob >= 0.5: return "High"
    if prob >= 0.2: return "Medium"
    return "Low"


def preprocess_symptoms(symptoms: str) -> str:
    """Convert symptom string to model input format"""
    # Handle comma-separated or newline-separated
    parts = symptoms.replace(",", "\n").split("\n")
    cleaned = [s.strip().lower() for s in parts if s.strip()]
    return " [SEP] ".join(cleaned)


# ── Routes ──

@app.get("/")
def root():
    return {
        "status" : "ok",
        "service": "AI DOC Rare Disease API",
        "version": "1.0.0",
        "models" : ["symptoms (TF-IDF + LR)"],
        "docs"   : "/docs",
    }

@app.get("/health")
def health():
    return {
        "status"   : "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": _tfidf is not None,
    }

@app.post("/predict/text")
async def predict_text(
    symptoms: str = Form(...),
    top_k: int    = Form(5),
):
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms cannot be empty")

    load_models()

    text    = preprocess_symptoms(symptoms)
    vec     = _tfidf.transform([text])
    proba   = _lr_model.predict_proba(vec)[0]
    k       = min(int(top_k), len(proba))
    top_idx = np.argsort(proba)[-k:][::-1]

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

    _history.append({
        "timestamp"  : datetime.utcnow().isoformat(),
        "symptoms"   : symptoms,
        "predictions": predictions,
        "mode"       : "symptoms_only",
    })

    return {
        "predictions": predictions,
        "mode"       : "symptoms_only",
        "input_text" : text,
    }

@app.post("/predict")
async def predict_multimodal(
    symptoms: str                = Form(...),
    top_k: int                   = Form(5),
    image: Optional[UploadFile]  = File(None),
):
    """
    Multimodal endpoint.
    Currently uses symptoms model only.
    Image model will be integrated after training completes.
    """
    result = await predict_text(symptoms=symptoms, top_k=top_k)

    if image:
        result["image_received"] = True
        result["image_filename"] = image.filename
        result["note"] = "Image model training in progress. Using symptoms model only."

    return result

@app.get("/analytics")
def analytics():
    total = len(_history)
    high  = sum(
        1 for h in _history
        if (h.get("predictions") or [{}])[0].get("confidence") == "High"
    )
    unique_diseases = len(set(
        (h.get("predictions") or [{}])[0].get("orpha_code", "")
        for h in _history
    ))
    return {
        "total_predictions" : total,
        "high_confidence"   : high,
        "unique_diseases"   : unique_diseases,
        "diseases_covered"  : 49,
        "models_active"     : 1,
        "model_accuracy"    : {
            "top1": 34.06,
            "top3": 52.92,
            "top5": 62.67,
            "f1"  : 0.3218,
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