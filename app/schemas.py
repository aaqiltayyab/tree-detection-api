"""
Pydantic response models — the contract this API exposes to callers.
Adding a new AI feature means adding a new response schema here.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """
    tree_detected    – True when a tree-class box clears the confidence threshold.
    confidence       – Best tree confidence (0.0 when no tree found).
    message          – Human-readable note; present only when tree_detected=False.
    label            – Winning class label e.g. "tree"; present only on detection.
    all_detections   – Every box the model saw e.g. ["car(0.91)", "bench(0.62)"].
                       Useful for debugging when tree_detected=False.
    """

    tree_detected:   bool
    confidence:      float = Field(..., ge=0.0, le=1.0)
    message:         Optional[str]       = None
    label:           Optional[str]       = None
    all_detections:  Optional[list[str]] = None


class ErrorDetail(BaseModel):
    detail: str
