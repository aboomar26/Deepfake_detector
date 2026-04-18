"""
Image preprocessing helpers.

Preprocessing is kept separate from the model so it can be reused
across the image router, the video router, and future endpoints.
"""

from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import settings

# ---------------------------------------------------------------------------
# Transform pipelines (one per model, mirrors the val/test transforms
# from each notebook — NO training augmentations at inference time)
# ---------------------------------------------------------------------------

_v1_transform = T.Compose(
    [
        T.Resize((settings.V1_IMG_SIZE, settings.V1_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(settings.NORM_MEAN, settings.NORM_STD),
    ]
)

_v2_transform = T.Compose(
    [
        T.Resize((settings.V2_IMG_SIZE, settings.V2_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(settings.NORM_MEAN, settings.NORM_STD),
    ]
)

# TTA transforms from Notebook 3 (used optionally for higher accuracy)
_v2_tta_transforms = [
    _v2_transform,
    T.Compose(
        [
            T.Resize((settings.V2_IMG_SIZE, settings.V2_IMG_SIZE)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(settings.NORM_MEAN, settings.NORM_STD),
        ]
    ),
    T.Compose(
        [
            T.Resize(
                (
                    int(settings.V2_IMG_SIZE * 1.1),
                    int(settings.V2_IMG_SIZE * 1.1),
                )
            ),
            T.CenterCrop(settings.V2_IMG_SIZE),
            T.ToTensor(),
            T.Normalize(settings.NORM_MEAN, settings.NORM_STD),
        ]
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pil_image(data: bytes) -> Image.Image:
    """Decode raw bytes → RGB PIL Image."""
    return Image.open(BytesIO(data)).convert("RGB")


def preprocess_for_v1(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for Model V1 (EfficientNet-B0, 128×128).
    Returns shape: (1, 3, 128, 128)
    """
    return _v1_transform(image).unsqueeze(0)


def preprocess_for_v2(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for Model V2 (EfficientNet-B3, 224×224).
    Returns shape: (1, 3, 224, 224)
    """
    return _v2_transform(image).unsqueeze(0)


def preprocess_frames_for_v1(frames: list[Image.Image]) -> torch.Tensor:
    """
    Stack multiple PIL frames into a single batch tensor for Model V1.
    Returns shape: (N, 3, 128, 128)
    """
    return torch.stack([_v1_transform(f) for f in frames])


def preprocess_frames_for_v2(
    frames: list[Image.Image], use_tta: bool = False
) -> torch.Tensor:
    """
    Stack frames for Model V2.  When use_tta=True each frame is passed
    through all three TTA transforms and the tensors are concatenated,
    so the returned batch is 3× larger.
    Returns shape: (N, 3, 224, 224)  or  (3N, 3, 224, 224) with TTA
    """
    if use_tta:
        tensors = []
        for tfm in _v2_tta_transforms:
            tensors.extend([tfm(f) for f in frames])
        return torch.stack(tensors)

    return torch.stack([_v2_transform(f) for f in frames])
