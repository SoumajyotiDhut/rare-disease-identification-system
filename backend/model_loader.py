import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
import pickle
import numpy as np
import os

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes,
                 text_dim=768, img_dim=2048,
                 fusion_dim=512, dropout=0.4):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2")

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(
            *list(backbone.children())[:-1])

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.img_proj  = nn.Linear(img_dim,  fusion_dim)

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        text_out  = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_feat)

        img_feat  = self.resnet(images).squeeze(-1).squeeze(-1)
        img_proj  = self.img_proj(img_feat)

        fused  = torch.cat([text_proj, img_proj], dim=1)
        return self.fusion(fused)


def load_models(models_dir="models"):
    device = torch.device("cpu")

    # Load checkpoint
    model_path = os.path.join(models_dir, "fusion_model.pt")
    torch.serialization.add_safe_globals([np.ndarray])
    checkpoint = torch.load(
        model_path, map_location=device,
        weights_only=False)

    NUM_CLASSES = checkpoint['num_classes']
    label_remap = checkpoint['label_remap']
    reverse_remap = checkpoint['reverse_remap']

    # Build + load model
    model = MultimodalFusionModel(NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.2")

    # Load label encoder
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"✓ Models loaded")
    print(f"  Classes     : {NUM_CLASSES}")

    return model, tokenizer, le, label_remap, reverse_remap, device