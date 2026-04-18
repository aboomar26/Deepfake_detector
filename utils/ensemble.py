"""
Ensemble helpers.

Combines predictions from multiple models into a single aggregated result.
Uses simple probability averaging (soft ensemble) as the default strategy,
which is the most robust approach when models are calibrated similarly.
"""

from typing import Optional


def ensemble_image(
    v1_prob_fake: float,
    v2_prob_fake: float,
    weights: tuple[float, float] = (0.4, 0.6),
) -> dict:
    """
    Soft ensemble of two image/video model outputs.

    Model V2 (EfficientNet-B3) is given slightly more weight because it
    is the improved model (NB3 vs NB1). Adjust `weights` as needed.

    Returns a dict with:
      - prediction:   "real" or "fake"
      - confidence:   max(prob_real, prob_fake) of the ensemble
      - prob_real:    ensemble real probability
      - prob_fake:    ensemble fake probability
    """
    w1, w2 = weights
    ens_fake = w1 * v1_prob_fake + w2 * v2_prob_fake
    ens_real = 1.0 - ens_fake
    pred = "fake" if ens_fake > 0.5 else "real"
    confidence = max(ens_real, ens_fake)
    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "prob_real": round(ens_real, 4),
        "prob_fake": round(ens_fake, 4),
        "weights_used": {"model_v1": w1, "model_v2": w2},
    }


def ensemble_video(
    v1_prob_fake: float,
    v2_prob_fake: float,
    audio_prob_fake: Optional[float],
    weights: tuple[float, float, float] = (0.25, 0.35, 0.40),
) -> dict:
    """
    Soft ensemble of two visual models + audio model for video.

    When audio is unavailable (video has no audio track), the visual
    weights are renormalized to sum to 1.

    Default weights give audio the most influence (0.40) because it is
    an independent modality and deepfake audio is often the weak point.
    """
    w_v1, w_v2, w_audio = weights

    if audio_prob_fake is None:
        # No audio track — renormalise visual weights
        total = w_v1 + w_v2
        w_v1 /= total
        w_v2 /= total
        w_audio = 0.0
        ens_fake = w_v1 * v1_prob_fake + w_v2 * v2_prob_fake
    else:
        ens_fake = (
            w_v1 * v1_prob_fake
            + w_v2 * v2_prob_fake
            + w_audio * audio_prob_fake
        )

    ens_real = 1.0 - ens_fake
    pred = "fake" if ens_fake > 0.5 else "real"
    confidence = max(ens_real, ens_fake)
    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "prob_real": round(ens_real, 4),
        "prob_fake": round(ens_fake, 4),
        "audio_included": audio_prob_fake is not None,
    }
