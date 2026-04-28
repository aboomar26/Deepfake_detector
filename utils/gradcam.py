"""
gradcam.py
==========
Production-ready Grad-CAM implementation for EfficientNet models.

Design principles:
  - Hooks are registered and removed inside a context manager — zero leaks.
  - The class is stateless between calls; it does NOT cache activations as
    instance attributes that survive the request.
  - Works identically on CPU and CUDA.
  - Does NOT modify or re-wrap the model in any way.
  - A single GradCAM instance can be reused across multiple requests
    (hooks are re-registered fresh for every generate() call).

Usage:
    gradcam = GradCAM(model, target_layer=model.features[-1])
    overlay_bytes = gradcam.generate(tensor, class_idx, original_pil_image)
"""

import base64
import io
import logging
from contextlib import contextmanager
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for any CNN.

    Parameters
    ----------
    model       : nn.Module — the *already-loaded, eval-mode* model.
                  Must NOT be modified (no .train() calls here).
    target_layer: nn.Module — the convolutional layer whose activations
                  are used.  For EfficientNet this is model.features[-1].
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

    # ------------------------------------------------------------------
    # Context manager: register → yield → remove hooks
    # ------------------------------------------------------------------
    @contextmanager
    def _hook_context(self):
        """
        Temporarily attach forward and backward hooks to the target layer.
        Hooks are ALWAYS removed in the finally block, preventing leaks
        even if an exception is raised mid-inference.
        """
        activations: list[torch.Tensor] = []
        gradients:   list[torch.Tensor] = []

        def fwd_hook(_module, _input, output):
            # Detach to avoid keeping the full computation graph
            activations.append(output.detach())

        def bwd_hook(_module, _grad_input, grad_output):
            # grad_output[0] is the gradient w.r.t. the layer's output
            gradients.append(grad_output[0].detach())

        fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

        try:
            yield activations, gradients
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    # ------------------------------------------------------------------
    # Core Grad-CAM computation
    # ------------------------------------------------------------------
    def _compute_cam(
        self,
        tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Run a single forward + backward pass with gradient tracking and
        compute the raw (unscaled) CAM heatmap.

        Returns a float32 numpy array shaped (H, W) in [0, 1].
        """
        with self._hook_context() as (activations, gradients):
            # --- Forward pass WITH gradient tracking ---
            # model is in eval() mode; we temporarily enable grad only
            # for this call, without touching model.training flag.
            self.model.zero_grad()

            # Enable grad even though we're not in a training context.
            with torch.set_grad_enabled(True):
                output = self.model(tensor)          # (1, num_classes)
                score  = output[0, class_idx]        # scalar for target class
                score.backward()                     # fills gradients list

        if not activations or not gradients:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Verify that target_layer is part of the forward graph."
            )

        act  = activations[0].squeeze(0)   # (C, H, W)
        grad = gradients[0].squeeze(0)     # (C, H, W)

        # Global Average Pooling of gradients → importance weights
        weights = grad.mean(dim=(1, 2))    # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(act.shape[1:], device=act.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * act[i]

        # ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        tensor: torch.Tensor,
        class_idx: int,
        original_image: Image.Image,
        alpha: float = 0.4,
    ) -> str:
        """
        Generate a Grad-CAM heatmap overlay for the given input tensor
        and class index, blended onto the original PIL image.

        Parameters
        ----------
        tensor         : (1, 3, H, W) float tensor already on the model's device.
        class_idx      : Index of the class to explain (0 = real, 1 = fake).
        original_image : PIL.Image — the un-preprocessed original image,
                         used only for display purposes.
        alpha          : Opacity of the heatmap overlay (0 = no overlay, 1 = full).

        Returns
        -------
        str — Base64-encoded PNG image of the overlay (no data: prefix).
        """
        try:
            cam = self._compute_cam(tensor, class_idx)
        except Exception as exc:
            logger.warning(f"Grad-CAM computation failed: {exc}")
            raise

        overlay = self._build_overlay(cam, original_image, alpha)
        return self._to_base64(overlay)

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_overlay(
        cam: np.ndarray,
        original_image: Image.Image,
        alpha: float,
    ) -> np.ndarray:
        """
        Resize the CAM to match the original image, colorize with JET,
        and blend with the original.

        Returns a uint8 BGR numpy array (H, W, 3).
        """
        target_w, target_h = original_image.size          # PIL: (width, height)

        # 1. Resize CAM to match original image dimensions
        cam_resized = cv2.resize(
            cam,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. Scale to uint8
        cam_uint8 = (cam_resized * 255).astype(np.uint8)

        # 3. Apply JET colormap → BGR uint8
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # 4. Convert original PIL image to BGR numpy
        orig_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

        # 5. Ensure same size (safety guard for edge cases)
        if heatmap_bgr.shape != orig_bgr.shape:
            heatmap_bgr = cv2.resize(
                heatmap_bgr,
                (orig_bgr.shape[1], orig_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 6. Weighted blend: alpha × heatmap + (1-alpha) × original
        overlay = cv2.addWeighted(heatmap_bgr, alpha, orig_bgr, 1.0 - alpha, 0)
        return overlay

    @staticmethod
    def _to_base64(image_bgr: np.ndarray) -> str:
        """Encode a BGR numpy image as a base64 PNG string."""
        success, buffer = cv2.imencode(".png", image_bgr)
        if not success:
            raise RuntimeError("cv2.imencode failed to encode the overlay image.")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Helper: resolve the correct target layer for EfficientNet (timm)
# ---------------------------------------------------------------------------

def get_efficientnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the last convolutional block of a timm EfficientNet model.

    timm EfficientNet models expose their MBConv blocks through
    model.blocks (a Sequential of blocks).  The last block is the
    deepest feature extractor before global average pooling, making
    it the canonical Grad-CAM target layer.

    Fallback: if model.blocks is unavailable (future timm refactors),
    we try model.features[-1] as originally specified, then raise a
    clear error.
    """
    # Primary path (timm ≥ 0.6): model.blocks
    if hasattr(model, "blocks"):
        return model.blocks[-1]

    # Secondary path: model.features (older timm or torchvision wrappers)
    if hasattr(model, "features"):
        return model.features[-1]

    raise AttributeError(
        "Cannot locate the target conv layer on this model. "
        "Expected model.blocks or model.features to exist."
    )
