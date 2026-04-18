"""
Model architecture definitions.

Exact replicas of the classifier heads defined in the three notebooks.
Do NOT modify — these must match the saved .pth checkpoint structures.
"""

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Model 1 — Notebook 1
# EfficientNet-B0 with custom two-layer head (image & video detection)
# IMG_SIZE = 128
# ---------------------------------------------------------------------------

def build_model_v1(num_classes: int = 2) -> nn.Module:
    """
    EfficientNet-B0 head from Notebook 1.

    Classifier: Dropout(0.4) → Linear(1280→256) → ReLU
                → Dropout(0.3) → Linear(256→2)
    """
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Model 2 — Notebook 3 (v3 improved)
# EfficientNet-B3 with deep BatchNorm head (image & video detection)
# IMG_SIZE = 224
# ---------------------------------------------------------------------------

def build_model_v2(num_classes: int = 2) -> nn.Module:
    """
    EfficientNet-B3 head from Notebook 3 (v3 improved).

    Classifier: Dropout(0.4) → Linear(1536→512) → BN → GELU
                → Dropout(0.3) → Linear(512→128) → BN → GELU
                → Dropout(0.2) → Linear(128→2)
    """
    model = timm.create_model("efficientnet_b3", pretrained=False)
    in_features = model.classifier.in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Audio Model — Notebook 2
# EfficientNet-B0 backbone + BatchNorm head, trained on Mel-spectrograms
# ---------------------------------------------------------------------------

class AudioDeepfakeModel(nn.Module):
    """
    EfficientNet-B0 backbone + custom head from Notebook 2.

    The model consumes 3-channel (RGB-style) Mel-spectrogram images.
    Backbone output (1280-d) → BN → Dropout → Linear(256) → ReLU
                             → Dropout → Linear(2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,       # remove default head
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 1280

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, 1280)
        return self.classifier(feats)     # (B, 2)
