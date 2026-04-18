"""
/predict/image endpoint
=======================
Accepts an uploaded image file, runs it through both visual models,
and returns per-model predictions plus an optional ensemble result.
"""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from services.image_video_service import run_v1, run_v2
from utils.ensemble import ensemble_image
from utils.image_utils import load_pil_image

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp", "image/gif",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class SingleModelPrediction(BaseModel):
    prediction: str
    confidence: float
    prob_real: float
    prob_fake: float


class ImagePredictionResponse(BaseModel):
    model_v1: SingleModelPrediction   # EfficientNet-B0 (NB1)
    model_v2: SingleModelPrediction   # EfficientNet-B3 (NB3)
    ensemble: dict | None = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/image",
    response_model=ImagePredictionResponse,
    summary="Detect deepfake in an image",
    description=(
        "Runs the uploaded image through two independent models "
        "(EfficientNet-B0 from NB1 and EfficientNet-B3 from NB3). "
        "Returns individual predictions plus an optional weighted ensemble."
    ),
)
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WEBP)"),
    use_tta: bool = Query(
        False,
        description="Apply Test-Time Augmentation on Model V2 (3× slower, slightly more accurate)",
    ),
    include_ensemble: bool = Query(True, description="Include weighted ensemble result"),
):
    # --- Validation ---
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Accepted: {sorted(ALLOWED_MIME)}",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)/(1024**2):.1f} MB). Max 20 MB.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # --- Load image ---
    try:
        image = load_pil_image(raw)
    except Exception as exc:
        logger.exception("Failed to decode image")
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    frames = [image]  # single-element list; inference functions expect a list

    # --- Inference ---
    try:
        v1_result = run_v1(frames)
        v2_result = run_v2(frames, use_tta=use_tta)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # --- Build response ---
    ens = None
    if include_ensemble:
        ens = ensemble_image(
            v1_prob_fake=v1_result.prob_fake,
            v2_prob_fake=v2_result.prob_fake,
        )

    return ImagePredictionResponse(
        model_v1=SingleModelPrediction(**v1_result._asdict()),
        model_v2=SingleModelPrediction(**v2_result._asdict()),
        ensemble=ens,
    )
