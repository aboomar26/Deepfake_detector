"""
Audio preprocessing helpers.

Exact port of the feature-extraction logic from Notebook 2:
  - load_audio()        → load + pad/trim to fixed duration
  - audio_to_melspec()  → log Mel-spectrogram → normalised 3-channel PIL image
  - preprocess_audio()  → full pipeline → torch tensor ready for the model
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import settings

# ---------------------------------------------------------------------------
# Eval transform (mirrors `eval_transform` from Notebook 2)
# ---------------------------------------------------------------------------

_audio_transform = T.Compose(
    [
        T.Resize((settings.AUDIO_IMG_SIZE, settings.AUDIO_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(settings.NORM_MEAN, settings.NORM_STD),
    ]
)


# ---------------------------------------------------------------------------
# Core functions (ported verbatim from Notebook 2)
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = None, duration: float = None) -> np.ndarray:
    """
    Load an audio file and fix its length to `duration` seconds.
    Pads with zeros if shorter; truncates if longer.
    Supports any format recognised by librosa (wav, flac, mp3, ogg …).
    """
    import librosa  # lazy import — not installed in all envs

    sr = sr or settings.AUDIO_SAMPLE_RATE
    duration = duration or settings.AUDIO_DURATION

    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    except Exception:
        y = np.zeros(int(sr * duration), dtype=np.float32)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32)


def audio_to_melspec(
    y: np.ndarray,
    sr: int = None,
    n_mels: int = None,
    n_fft: int = None,
    hop_length: int = None,
    img_size: int = None,
) -> Image.Image:
    """
    Convert a waveform → Log Mel-Spectrogram → 3-channel (RGB-style) PIL Image.

    Steps (exact mirror of Notebook 2):
      1. Compute Mel-spectrogram
      2. Convert to dB scale
      3. Normalise to [0, 255]
      4. Resize to img_size × img_size
      5. Replicate to 3 channels (greyscale → RGB)
    """
    import librosa

    sr = sr or settings.AUDIO_SAMPLE_RATE
    n_mels = n_mels or settings.AUDIO_N_MELS
    n_fft = n_fft or settings.AUDIO_N_FFT
    hop_length = hop_length or settings.AUDIO_HOP_LENGTH
    img_size = img_size or settings.AUDIO_IMG_SIZE

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

    # Normalise to [0, 255]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_db = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
    mel_db = (mel_db * 255).astype(np.uint8)

    # Resize and make RGB (3-channel)
    img = Image.fromarray(mel_db).resize((img_size, img_size))
    img_rgb = Image.merge("RGB", [img, img, img])
    return img_rgb


def preprocess_audio(raw_bytes: bytes) -> torch.Tensor:
    """
    Full pipeline: raw audio bytes → (1, 3, H, W) tensor for the audio model.

    Writes the bytes to a temporary file so librosa can read it properly
    (librosa needs a seekable file-like object or a path).
    """
    # Determine format from the bytes magic (fallback to .wav)
    suffix = _guess_suffix(raw_bytes)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        y = load_audio(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    img = audio_to_melspec(y)
    tensor = _audio_transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def preprocess_audio_from_path(path: str) -> torch.Tensor:
    """Preprocess audio from an already-saved file path."""
    y = load_audio(path)
    img = audio_to_melspec(y)
    return _audio_transform(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAGIC = {
    b"fLaC": ".flac",
    b"RIFF": ".wav",
    b"OggS": ".ogg",
    b"\xff\xfb": ".mp3",
    b"\xff\xf3": ".mp3",
    b"\xff\xf2": ".mp3",
    b"ID3": ".mp3",
}


def _guess_suffix(data: bytes) -> str:
    for magic, ext in _MAGIC.items():
        if data[: len(magic)] == magic:
            return ext
    return ".wav"
