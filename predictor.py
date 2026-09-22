"""
predictor.py

Loads the fine-tuned ViT model and class names once at import time, and
exposes a predict(image_bytes) function used by the FastAPI app.
"""

import io
import json
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

CHECKPOINT = "google/vit-base-patch16-224"

# --- Load class names ---
with open("artifacts/class_names.json") as f:
    CLASS_NAMES = json.load(f)

# --- Load image processor + reconstruct model architecture ---
image_processor = ViTImageProcessor.from_pretrained(CHECKPOINT)

model = ViTForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True,
)

# --- Load fine-tuned weights (trained on GPU, loaded here on CPU) ---
model.load_state_dict(
    torch.load("artifacts/vit_brain_tumor.pt", map_location="cpu")
)
model.eval()


def predict(image_bytes: bytes) -> dict:
    """
    Accepts raw image bytes, runs the fine-tuned ViT model, and returns
    the predicted class label and confidence score.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }
