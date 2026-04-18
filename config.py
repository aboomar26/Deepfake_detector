"""
Application configuration.
Override any value with the corresponding environment variable.

Example:
  export MODEL_V1_PATH=/data/models/best_deepfake_model.pth
  export AUDIO_MODEL_PATH=/data/models/best_audio_deepfake_model.pth
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # ── Model checkpoint paths ──────────────────────────────────────────
    # Model 1: EfficientNet-B0  trained in Notebook 1 (image / video)
    MODEL_V1_PATH: str = os.getenv(
        "MODEL_V1_PATH", "checkpoints/best_deepfake_model.pth"
    )
    # Model 2: EfficientNet-B3  trained in Notebook 3 (image / video, improved)
    MODEL_V2_PATH: str = os.getenv(
        "MODEL_V2_PATH", "checkpoints/best_deepfake_model_v3.pth"
    )
    # Audio model: EfficientNet-B0 on Mel-spectrograms, Notebook 2
    AUDIO_MODEL_PATH: str = os.getenv(
        "AUDIO_MODEL_PATH", "checkpoints/best_audio_deepfake_model.pth"
    )

    # ── Model V1 inference settings (Notebook 1) ───────────────────────
    V1_IMG_SIZE: int = 128

    # ── Model V2 inference settings (Notebook 3) ───────────────────────
    V2_IMG_SIZE: int = 224

    # ── Audio model settings (Notebook 2) ──────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16_000
    AUDIO_DURATION: float = 4.0     # seconds to keep / pad
    AUDIO_N_MELS: int = 128
    AUDIO_N_FFT: int = 1024
    AUDIO_HOP_LENGTH: int = 512
    AUDIO_IMG_SIZE: int = 128       # spectrogram image size

    # ── Video processing ────────────────────────────────────────────────
    VIDEO_MAX_FRAMES: int = 16      # frames to sample per video
    VIDEO_BATCH_SIZE: int = 8       # frames sent to GPU at once

    # ── ImageNet normalisation (shared by all three models) ─────────────
    NORM_MEAN: tuple = (0.485, 0.456, 0.406)
    NORM_STD: tuple = (0.229, 0.224, 0.225)


settings = Settings()
