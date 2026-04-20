"""
Model lifecycle — loading, warm-up, and inference.

Uses OpenAI CLIP (openai/clip-vit-base-patch32) for zero-shot tree detection.
CLIP understands natural language, so it correctly identifies trees even when
people, buildings, or other objects share the frame — no fine-tuning needed.

Model is auto-downloaded from HuggingFace on first run (~600 MB, free, no API key).
"""

import io
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import Settings

logger = logging.getLogger(__name__)

#  Module-level singletons (loaded once at startup) 
_model:     Optional[CLIPModel]     = None
_processor: Optional[CLIPProcessor] = None


def load_model(settings: Settings) -> tuple[CLIPModel, CLIPProcessor]:
    """
    Download (first run) or load CLIP from local HuggingFace cache.
    Sets the model to eval mode and moves it to GPU if available.
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    logger.info("Loading CLIP model: %s  (downloads ~600 MB on first run)", settings.CLIP_MODEL)

    _processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
    _model     = CLIPModel.from_pretrained(settings.CLIP_MODEL)
    _model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)

    logger.info("CLIP ready on device=%s", device)
    return _model, _processor


def get_model() -> tuple[CLIPModel, CLIPProcessor]:
    """FastAPI dependency — returns already-loaded singletons."""
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Was load_model() called at startup?")
    return _model, _processor


#  Inference 

def run_tree_inference(
    image_bytes: bytes,
    model_and_processor: tuple[CLIPModel, CLIPProcessor],
    settings: Settings,
) -> dict:
    """
    Run CLIP zero-shot inference on raw image bytes.

    Strategy:
      • Build a candidate list: positive prompts ("a photo of a tree", ...)
        followed by negative prompts ("a photo with no trees", ...).
      • Compute softmax probabilities over all candidates.
      • tree_detected = True if the best positive-prompt probability ≥ threshold.

    Returns:
        tree_detected   – bool
        confidence      – float, best positive-prompt probability (0–1)
        label           – the winning positive prompt text, or None
        all_detections  – list of "prompt(score)" for every candidate
    """
    model, processor = model_and_processor
    image = _decode_image(image_bytes)

    positive = settings.positive_prompts
    negative = settings.negative_prompts
    all_labels = positive + negative

    device = next(model.parameters()).device

    inputs = processor(
        text=all_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image shape: [1, num_labels]
        probs = outputs.logits_per_image.softmax(dim=-1)[0]   # shape: [num_labels]

    # Score each label
    scored = [(label, float(probs[i])) for i, label in enumerate(all_labels)]
    all_detections = [f"{label}({score:.2f})" for label, score in scored]

    # Best positive prompt
    best_label: Optional[str] = None
    best_conf:  float = 0.0
    for label, score in scored[:len(positive)]:
        if score > best_conf:
            best_conf  = score
            best_label = label

    tree_found = best_conf >= settings.CONFIDENCE_THRESHOLD

    if tree_found:
        logger.info(
            "RESULT ✓ tree_detected=True   conf=%.3f  prompt=%r",
            best_conf, best_label,
        )
    else:
        logger.info(
            "RESULT ✗ tree_detected=False  best_positive=%.3f  threshold=%.2f  scores=%s",
            best_conf, settings.CONFIDENCE_THRESHOLD, all_detections,
        )

    return {
        "tree_detected":  tree_found,
        "confidence":     round(best_conf, 4),
        "label":          best_label if tree_found else None,
        "all_detections": all_detections,
    }


#  Helpers 

def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes → RGB PIL Image."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc
