"""
API routing layer — only HTTP concerns live here.
Business logic stays in model.py; config stays in config.py.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from transformers import CLIPModel, CLIPProcessor

from app.config import Settings, get_settings
from app.model import get_model, run_tree_inference
from app.schemas import DetectionResult, ErrorDetail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tree-detection"])


#  Endpoint 

@router.post(
    "/detect",
    response_model=DetectionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect whether a tree is present in an uploaded image",
    responses={
        400: {"model": ErrorDetail, "description": "Invalid or missing image"},
        413: {"model": ErrorDetail, "description": "File too large"},
        422: {"model": ErrorDetail, "description": "Unsupported media type"},
        500: {"model": ErrorDetail, "description": "Inference failure"},
    },
)
async def detect_tree(
    file:     Annotated[UploadFile, File(description="JPG, PNG, or WEBP image")],
    model_and_processor: Annotated[tuple[CLIPModel, CLIPProcessor], Depends(get_model)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DetectionResult:
    """
    Upload an image — get back whether a tree was detected.

    Uses CLIP zero-shot classification: no YOLO class lists, no fine-tuning.
    Works even when people, cars, or buildings share the frame with a tree.
    `all_detections` shows the probability for every candidate prompt.
    """
    t0 = time.perf_counter()

    #  Validate content type 
    allowed = settings.allowed_content_type_list
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted: {', '.join(allowed)}"
            ),
        )

    #  Read & size-check 
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(image_bytes) / 1_048_576:.1f} MB exceeds the "
                f"{settings.MAX_UPLOAD_BYTES // 1_048_576} MB limit."
            ),
        )

    #  Inference 
    try:
        result = run_tree_inference(image_bytes, model_and_processor, settings)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected inference error for file '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again.",
        ) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "REQUEST  file=%-30s  size=%7dB  tree=%-5s  conf=%.3f  latency=%.1fms",
        file.filename, len(image_bytes),
        result["tree_detected"], result["confidence"], elapsed_ms,
    )

    #  Build response 
    if not result["tree_detected"]:
        return DetectionResult(
            tree_detected=False,
            confidence=0.0,
            message="No tree found. Please upload a photo with a tree.",
            all_detections=result["all_detections"],
        )

    return DetectionResult(
        tree_detected=True,
        confidence=result["confidence"],
        label=result["label"],
        all_detections=result["all_detections"],
    )


#  Health check 

@router.get("/health", tags=["ops"], include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
