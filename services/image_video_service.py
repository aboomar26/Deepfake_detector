"""
ImageVideoInferenceService
==========================
Runs inference with Model V1 (EfficientNet-B0, NB1) and Model V2
(EfficientNet-B3, NB3) on batches of PIL Images.

Inference is separated from routing so both the image and video
endpoints can reuse the same logic without duplication.
"""

import logging
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import settings
from services.model_registry import ModelRegistry
from utils.image_utils import (
    preprocess_frames_for_v1,
    preprocess_frames_for_v2,
)

logger = logging.getLogger(__name__)


class FrameResult(NamedTuple):
    """Per-frame (or per-image) prediction from a single model."""
    prediction: str      # "real" or "fake"
    confidence: float    # 0.0 – 1.0
    prob_real: float
    prob_fake: float


class ModelResult(NamedTuple):
    """Aggregated result across all frames for one model."""
    prediction: str
    confidence: float
    prob_real: float
    prob_fake: float


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_v1(frames: list[Image.Image]) -> ModelResult:
    """
    Run Model V1 (EfficientNet-B0, IMG_SIZE=128) on a list of frames.

    For a single image pass a one-element list.
    For video, frame-level probabilities are averaged (soft voting,
    matching Notebook 1's test evaluation logic).
    """
    registry = ModelRegistry.get_instance()
    model = registry.model_v1
    device = registry.device

    batch = preprocess_frames_for_v1(frames).to(device)
    probs = _batch_softmax(model, batch, settings.VIDEO_BATCH_SIZE)

    return _aggregate(probs)


@torch.no_grad()
def run_v2(frames: list[Image.Image], use_tta: bool = False) -> ModelResult:
    """
    Run Model V2 (EfficientNet-B3, IMG_SIZE=224) on a list of frames.

    When use_tta=True the three TTA transforms from Notebook 3 are applied
    and their softmax outputs are averaged before aggregation.
    """
    registry = ModelRegistry.get_instance()
    model = registry.model_v2
    device = registry.device

    batch = preprocess_frames_for_v2(frames, use_tta=use_tta).to(device)
    probs = _batch_softmax(model, batch, settings.VIDEO_BATCH_SIZE)

    if use_tta:
        # Reshape (3*N, 2) → (N, 3, 2) then mean over TTA dimension
        n = len(frames)
        probs = probs.reshape(3, n, 2).mean(dim=0)  # (N, 2)

    return _aggregate(probs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_softmax(
    model: torch.nn.Module,
    batch: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """
    Run model in chunks to avoid OOM on large batches.
    Returns softmax probabilities shape (N, 2).
    """
    all_probs = []
    for i in range(0, len(batch), chunk):
        sub = batch[i : i + chunk]
        logits = model(sub)
        all_probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(all_probs, dim=0)  # (N, 2)


def _aggregate(probs: torch.Tensor) -> ModelResult:
    """
    Soft-vote across N frames: average probabilities then threshold at 0.5.
    Mirrors the soft-voting strategy from Notebook 3's video_level_eval().
    """
    avg = probs.mean(dim=0)          # (2,)
    prob_real = avg[0].item()
    prob_fake = avg[1].item()
    pred = "fake" if prob_fake > 0.5 else "real"
    confidence = max(prob_real, prob_fake)
    return ModelResult(
        prediction=pred,
        confidence=round(confidence, 4),
        prob_real=round(prob_real, 4),
        prob_fake=round(prob_fake, 4),
    )
