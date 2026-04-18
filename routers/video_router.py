"""
/predict/video endpoint
=======================
Accepts a video file and runs a full multi-modal analysis:
  1. Extract N frames  → Model V1 (EfficientNet-B0)
  2. Extract N frames  → Model V2 (EfficientNet-B3, optional TTA)
  3. Extract audio     → Audio Model (EfficientNet-B0 on Mel-spectrogram)
  4. Combine all three into a weighted ensemble decision

Frame extraction: evenly-spaced sampling (configurable N).
Audio extraction: ffmpeg rips the audio track to WAV, then the same
                  pipeline as /predict/audio is applied.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from services.audio_service import predict_audio
from services.image_video_service import run_v1, run_v2
from utils.ensemble import ensemble_video
from utils.video_utils import extract_audio, extract_frames

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/x-matroska",
    "video/webm", "video/ogg",
    "application/octet-stream",
}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class VisualModelResult(BaseModel):
    prediction: str
    confidence: float
    prob_real: float
    prob_fake: float
    frames_analysed: int


class AudioModelResult(BaseModel):
    prediction: str
    confidence: float
    prob_real: float
    prob_fake: float
    available: bool


class VideoPredictionResponse(BaseModel):
    model_v1: VisualModelResult
    model_v2: VisualModelResult
    audio: AudioModelResult
    ensemble: dict | None = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/video",
    response_model=VideoPredictionResponse,
    summary="Detect deepfake in a video (visual + audio)",
    description=(
        "Extracts frames and audio from a video then runs three models: "
        "EfficientNet-B0 (NB1), EfficientNet-B3 (NB3), and the audio model (NB2). "
        "Returns per-model predictions and a combined ensemble decision."
    ),
)
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MKV, MOV …)"),
    n_frames: int = Query(
        16,
        ge=1,
        le=64,
        description="Number of frames to sample evenly from the video",
    ),
    use_tta: bool = Query(
        False,
        description="Apply TTA on Model V2 (3× slower, slightly more accurate)",
    ),
    include_ensemble: bool = Query(True, description="Include weighted ensemble result"),
):
    # --- Validation ---
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)/(1024**2):.1f} MB). Max 500 MB.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # --- Extract frames (CPU-bound, run in thread pool) ---
    try:
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            None, extract_frames, raw, n_frames
        )
    except Exception as exc:
        logger.exception("Frame extraction failed")
        raise HTTPException(
            status_code=422, detail=f"Could not extract frames: {exc}"
        )

    # --- Visual inference ---
    try:
        v1_result = run_v1(frames)
        v2_result = run_v2(frames, use_tta=use_tta)
    except Exception as exc:
        logger.exception("Visual inference failed")
        raise HTTPException(status_code=500, detail=f"Visual inference error: {exc}")

    # --- Extract audio (may return None if no audio track / no ffmpeg) ---
    audio_bytes: Optional[bytes] = await loop.run_in_executor(
        None, extract_audio, raw
    )

    # --- Audio inference ---
    audio_result = None
    audio_available = False
    if audio_bytes:
        try:
            audio_result = predict_audio(audio_bytes)
            audio_available = True
        except Exception as exc:
            logger.warning(f"Audio inference failed (non-fatal): {exc}")

    # --- Build response ---
    audio_section = AudioModelResult(
        prediction=audio_result.prediction if audio_result else "n/a",
        confidence=audio_result.confidence if audio_result else 0.0,
        prob_real=audio_result.prob_real if audio_result else 0.0,
        prob_fake=audio_result.prob_fake if audio_result else 0.0,
        available=audio_available,
    )

    ens = None
    if include_ensemble:
        ens = ensemble_video(
            v1_prob_fake=v1_result.prob_fake,
            v2_prob_fake=v2_result.prob_fake,
            audio_prob_fake=audio_result.prob_fake if audio_result else None,
        )

    return VideoPredictionResponse(
        model_v1=VisualModelResult(**v1_result._asdict(), frames_analysed=len(frames)),
        model_v2=VisualModelResult(**v2_result._asdict(), frames_analysed=len(frames)),
        audio=audio_section,
        ensemble=ens,
    )
