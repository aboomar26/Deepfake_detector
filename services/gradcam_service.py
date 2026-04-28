"""
gradcam_service.py
==================
Thin service layer that wires the ModelRegistry with the GradCAM utility.

Design decisions:
  - One GradCAM instance per model, created lazily on first use and
    cached on the registry (no second model load, no duplicate objects).
  - Completely separate from run_v1 / run_v2 — those functions are
    untouched and still work exactly as before.
  - The gradient-enabled forward pass is isolated to generate_gradcam();
    normal inference still uses @torch.no_grad() via the existing service.
  - Only called when prediction == "fake", as specified.
"""

import logging
from typing import Optional

import torch
from PIL import Image

from services.model_registry import ModelRegistry
from utils.gradcam import GradCAM, get_efficientnet_target_layer
from utils.image_utils import preprocess_for_v1, preprocess_for_v2

logger = logging.getLogger(__name__)

# Class index for FAKE (matches training: label 1 = fake)
FAKE_CLASS_IDX = 1


# ---------------------------------------------------------------------------
# Lazy GradCAM instance cache — attached to the registry at first call
# ---------------------------------------------------------------------------

def _get_gradcam_v1() -> GradCAM:
    """Return (and lazily create) the GradCAM instance for Model V1."""
    registry = ModelRegistry.get_instance()
    if not hasattr(registry, "_gradcam_v1") or registry._gradcam_v1 is None:
        target = get_efficientnet_target_layer(registry.model_v1)
        registry._gradcam_v1 = GradCAM(registry.model_v1, target)
        logger.info("GradCAM for Model V1 initialized (target: %s)", target.__class__.__name__)
    return registry._gradcam_v1


def _get_gradcam_v2() -> GradCAM:
    """Return (and lazily create) the GradCAM instance for Model V2."""
    registry = ModelRegistry.get_instance()
    if not hasattr(registry, "_gradcam_v2") or registry._gradcam_v2 is None:
        target = get_efficientnet_target_layer(registry.model_v2)
        registry._gradcam_v2 = GradCAM(registry.model_v2, target)
        logger.info("GradCAM for Model V2 initialized (target: %s)", target.__class__.__name__)
    return registry._gradcam_v2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam_v1(
    image: Image.Image,
    alpha: float = 0.4,
) -> Optional[str]:
    """
    Generate a Grad-CAM overlay using Model V1 (EfficientNet-B0, 128×128).

    Parameters
    ----------
    image : Original PIL image (any size — will be preprocessed internally).
    alpha : Heatmap blend opacity.

    Returns
    -------
    Base64 PNG string, or None if generation fails.
    """
    return _generate(image, _get_gradcam_v1, preprocess_for_v1, alpha, "V1")


def generate_gradcam_v2(
    image: Image.Image,
    alpha: float = 0.4,
) -> Optional[str]:
    """
    Generate a Grad-CAM overlay using Model V2 (EfficientNet-B3, 224×224).

    Parameters
    ----------
    image : Original PIL image (any size — will be preprocessed internally).
    alpha : Heatmap blend opacity.

    Returns
    -------
    Base64 PNG string, or None if generation fails.
    """
    return _generate(image, _get_gradcam_v2, preprocess_for_v2, alpha, "V2")


# ---------------------------------------------------------------------------
# Internal shared logic
# ---------------------------------------------------------------------------

def _generate(
    image: Image.Image,
    gradcam_getter,
    preprocess_fn,
    alpha: float,
    label: str,
) -> Optional[str]:
    """
    Shared implementation for both models.

    Steps:
      1. Preprocess image to (1, 3, H, W) tensor on the correct device.
      2. Call GradCAM.generate() — this runs its own grad-enabled forward
         pass (completely separate from the no_grad inference pass that
         already happened to produce the prediction).
      3. Return base64 PNG or None on failure.
    """
    registry = ModelRegistry.get_instance()
    device = registry.device

    try:
        # Build the input tensor (same preprocessing as inference)
        tensor = preprocess_fn(image).to(device)   # (1, 3, H, W)

        gradcam = gradcam_getter()
        b64 = gradcam.generate(
            tensor=tensor,
            class_idx=FAKE_CLASS_IDX,
            original_image=image,
            alpha=alpha,
        )
        logger.debug("Grad-CAM %s generated successfully (%d bytes b64)", label, len(b64))
        return b64

    except Exception as exc:
        # Never let Grad-CAM failure propagate to the caller —
        # the prediction has already been computed and should be returned.
        logger.warning("Grad-CAM %s generation failed (non-fatal): %s", label, exc)
        return None
