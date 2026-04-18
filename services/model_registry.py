"""
ModelRegistry
=============
Singleton that loads every model once at application startup and holds
them in memory for the lifetime of the process.

Expected checkpoint files (configure via environment variables or config.py):
  - MODEL_V1_PATH  →  best_deepfake_model.pth       (EfficientNet-B0, NB1)
  - MODEL_V2_PATH  →  best_deepfake_model_v3.pth    (EfficientNet-B3, NB3)
  - AUDIO_MODEL_PATH → best_audio_deepfake_model.pth (EfficientNet-B0, NB2)
"""

import logging
import threading
from pathlib import Path
import numpy as np
import torch

from models.architectures import build_model_v1, build_model_v2, AudioDeepfakeModel
from config import settings


# torch.serialization.add_safe_globals([np.core.multiarray.scalar]) ################

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe singleton model store."""

    _instance: "ModelRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_v1 = None   # EfficientNet-B0  (NB1)
        self.model_v2 = None   # EfficientNet-B3  (NB3)
        self.audio_model = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        logger.info(f"Device: {self.device}")
        self.model_v1 = self._load(
            build_model_v1(),
            settings.MODEL_V1_PATH,
            "Model V1 (EfficientNet-B0, NB1)",
        )
        self.model_v2 = self._load(
            build_model_v2(),
            settings.MODEL_V2_PATH,
            "Model V2 (EfficientNet-B3, NB3)",
        )
        self.audio_model = self._load(
            AudioDeepfakeModel(),
            settings.AUDIO_MODEL_PATH,
            "Audio Model (EfficientNet-B0, NB2)",
        )

    def unload_all(self) -> None:
        self.model_v1 = None
        self.model_v2 = None
        self.audio_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def are_loaded(self) -> bool:
        return all(
            m is not None
            for m in [self.model_v1, self.model_v2, self.audio_model]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self, model: torch.nn.Module, path: str, name: str) -> torch.nn.Module:
        p = Path(path)
        if not p.exists():
            logger.warning(
                f"{name}: checkpoint not found at '{path}'. "
                "Serving with random weights — predictions will be meaningless."
            )
            return model.to(self.device).eval()

        logger.info(f"Loading {name} from '{path}' …")
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Some checkpoints are full dicts (saved with extra metadata)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        logger.info(f"  ✓ {name} ready.")
        return model
