"""
/predict/audio endpoint
=======================
Accepts an uploaded audio file, converts it to a Mel-spectrogram image,
runs it through the audio model (EfficientNet-B0, Notebook 2), and
returns the prediction with confidence.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.audio_service import predict_audio

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME = {
    "audio/flac", "audio/x-flac",
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "application/ogg",
    "audio/webm",
    "audio/mp4",
    "application/octet-stream",  # generic binary — accept and let librosa decide
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class AudioPredictionResponse(BaseModel):
    prediction: str   # "real" or "fake"
    confidence: float
    prob_real: float
    prob_fake: float


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/audio",
    response_model=AudioPredictionResponse,
    summary="Detect AI-generated / deepfake audio",
    description=(
        "Converts the audio to a Log Mel-Spectrogram (128×128 RGB image) "
        "and passes it through an EfficientNet-B0 model trained on FLAC files. "
        "Accepts WAV, FLAC, MP3, OGG and most other formats supported by librosa."
    ),
)
async def predict_audio_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file (WAV, FLAC, MP3, OGG …)",
    ),
):
    # --- Validation ---
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)/(1024**2):.1f} MB). Max 50 MB.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # --- Inference ---
    try:
        result = predict_audio(raw)
    except Exception as exc:
        logger.exception("Audio inference failed")
        raise HTTPException(status_code=500, detail=f"Audio inference error: {exc}")

    return AudioPredictionResponse(**result._asdict())
