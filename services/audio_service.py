"""
AudioInferenceService
=====================
Runs inference with the audio model (EfficientNet-B0, Notebook 2).

The predict_audio() function is a direct port of the `predict_audio`
helper defined in Cell 15 of Notebook 2.
"""

import logging
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from services.model_registry import ModelRegistry
from utils.audio_utils import preprocess_audio

logger = logging.getLogger(__name__)


class AudioResult(NamedTuple):
    prediction: str    # "real" or "fake"
    confidence: float  # 0.0 – 1.0
    prob_real: float
    prob_fake: float


@torch.no_grad()
def predict_audio(raw_bytes: bytes) -> AudioResult:
    """
    Predict whether an audio clip is real or AI-generated.

    Pipeline (Notebook 2, Cell 15):
      raw bytes
        → load_audio()       # resample + pad/trim to 4 s
        → audio_to_melspec() # log Mel-spectrogram → 128×128 RGB image
        → eval_transform()   # normalise
        → EfficientNet-B0 backbone + custom head
        → softmax → argmax → label + confidence
    """
    registry = ModelRegistry.get_instance()
    model = registry.audio_model
    device = registry.device

    # Preprocess: bytes → (1, 3, 128, 128) tensor
    tensor = preprocess_audio(raw_bytes).to(device)

    # Forward pass
    logits = model(tensor)                               # (1, 2)
    probs = F.softmax(logits, dim=1).cpu().squeeze(0)   # (2,)

    prob_real = probs[0].item()
    prob_fake = probs[1].item()
    pred = "fake" if prob_fake > 0.5 else "real"
    confidence = max(prob_real, prob_fake)

    return AudioResult(
        prediction=pred,
        confidence=round(confidence, 4),
        prob_real=round(prob_real, 4),
        prob_fake=round(prob_fake, 4),
    )
