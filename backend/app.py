from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import uvicorn
from model_loader import load_models

app = FastAPI(
    title="Rare Disease Identification API",
    description="Predicts Top-K rare diseases from symptoms and medical images",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models on startup
print("Loading models...")
model, tokenizer, le, label_remap, reverse_remap, device = load_models()
print("✓ Ready")

# Image transform
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])


@app.get("/")
def root():
    return {
        "message": "Rare Disease Identification API",
        "status" : "running",
        "endpoints": {
            "/predict"     : "POST — predict diseases from symptoms + image",
            "/predict/text": "POST — predict from symptoms only",
            "/health"      : "GET  — health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict")
async def predict(
    symptoms: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(default=5)
):
    """
    Predict Top-K diseases from symptoms + medical image

    - symptoms: comma-separated symptom list
    - image: medical image file (jpg/png)
    - top_k: number of diseases to return (default 5)
    """
    try:
        # ── Process symptoms ───────────────────────────
        symptom_list = [s.strip().lower()
                        for s in symptoms.split(',')]
        symptom_text = ' [SEP] '.join(symptom_list)

        enc = tokenizer(
            symptom_text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # ── Process image ──────────────────────────────
        img_bytes = await image.read()
        img       = Image.open(
            io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = img_transform(img).unsqueeze(0)

        # ── Run inference ──────────────────────────────
        with torch.no_grad():
            logits = model(
                enc['input_ids'].to(device),
                enc['attention_mask'].to(device),
                img_tensor.to(device)
            )
            probs = F.softmax(logits, dim=1)
            topk  = probs.topk(
                min(top_k, probs.size(1)), dim=1)

        # ── Build response ─────────────────────────────
        predictions = []
        for i in range(topk.indices.size(1)):
            label_idx    = topk.indices[0][i].item()
            prob         = topk.values[0][i].item()
            orig_label   = reverse_remap.get(
                label_idx, label_idx)
            try:
                disease = le.inverse_transform(
                    [orig_label])[0]
            except:
                disease = f"Disease_{orig_label}"

            predictions.append({
                "rank"       : i + 1,
                "disease"    : disease,
                "probability": round(prob * 100, 2),
                "confidence" : "High" if prob > 0.5
                               else "Medium" if prob > 0.2
                               else "Low"
            })

        return {
            "status"     : "success",
            "symptoms"   : symptom_list,
            "predictions": predictions,
            "top_k"      : top_k
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict/text")
async def predict_text(
    symptoms: str = Form(...),
    top_k: int = Form(default=5)
):
    """Predict using symptoms only — no image required"""
    try:
        symptom_list = [s.strip().lower()
                        for s in symptoms.split(',')]
        symptom_text = ' [SEP] '.join(symptom_list)

        enc = tokenizer(
            symptom_text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Use blank image for text-only prediction
        blank_img = torch.zeros(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(
                enc['input_ids'].to(device),
                enc['attention_mask'].to(device),
                blank_img.to(device)
            )
            probs = F.softmax(logits, dim=1)
            topk  = probs.topk(
                min(top_k, probs.size(1)), dim=1)

        predictions = []
        for i in range(topk.indices.size(1)):
            label_idx  = topk.indices[0][i].item()
            prob       = topk.values[0][i].item()
            orig_label = reverse_remap.get(
                label_idx, label_idx)
            try:
                disease = le.inverse_transform(
                    [orig_label])[0]
            except:
                disease = f"Disease_{orig_label}"

            predictions.append({
                "rank"       : i + 1,
                "disease"    : disease,
                "probability": round(prob * 100, 2),
                "confidence" : "High" if prob > 0.5
                               else "Medium" if prob > 0.2
                               else "Low"
            })

        return {
            "status"     : "success",
            "symptoms"   : symptom_list,
            "predictions": predictions,
            "top_k"      : top_k
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)