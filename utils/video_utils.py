"""
Video processing helpers.

Provides two capabilities:
  1. extract_frames()  — sample N evenly-spaced frames from a video file
  2. extract_audio()   — rip the audio track from a video into raw bytes

Both work entirely from bytes (suitable for file uploads) and use
temporary files under the hood so the caller never needs to touch the disk.
"""

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_frames(
    video_bytes: bytes,
    n_frames: int = None,
) -> list[Image.Image]:
    """
    Decode a video from raw bytes and return `n_frames` evenly-spaced
    RGB PIL Images.

    Strategy:
      - Write bytes to a temp file.
      - Count total frames via OpenCV.
      - Sample at uniform intervals.
      - Return PIL Images (ready for the transform pipelines).
    """
    n_frames = n_frames or settings.VIDEO_MAX_FRAMES

    suffix = _guess_video_suffix(video_bytes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames: list[Image.Image] = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video (format: {suffix})")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Some containers don't expose frame count — fall back to sequential read
            frames = _read_sequential(cap, n_frames)
        else:
            frames = _read_sampled(cap, total, n_frames)

        cap.release()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    logger.info(f"Extracted {len(frames)} frames from video.")
    return frames


def extract_audio(video_bytes: bytes) -> bytes | None:
    """
    Extract the audio track from a video using ffmpeg.

    Returns raw WAV bytes, or None if the video has no audio track or
    ffmpeg is not available.
    """
    suffix = _guess_video_suffix(video_bytes)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_vid:
        tmp_vid.write(video_bytes)
        vid_path = tmp_vid.name

    out_path = vid_path + "_audio.wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",              # overwrite output
                "-i", vid_path,
                "-vn",             # no video
                "-acodec", "pcm_s16le",
                "-ar", str(settings.AUDIO_SAMPLE_RATE),
                "-ac", "1",        # mono
                out_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            if "no audio" in stderr.lower() or "audio stream" in stderr.lower():
                logger.info("Video has no audio track — skipping audio prediction.")
            else:
                logger.warning(f"ffmpeg exited {result.returncode}: {stderr[:300]}")
            return None

        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            return None

        return Path(out_path).read_bytes()

    except FileNotFoundError:
        logger.warning("ffmpeg not found. Audio extraction skipped.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out. Audio extraction skipped.")
        return None
    finally:
        Path(vid_path).unlink(missing_ok=True)
        if Path(out_path).exists():
            Path(out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_sampled(cap: cv2.VideoCapture, total: int, n: int) -> list[Image.Image]:
    """Read n evenly-spaced frames from a video with known frame count."""
    indices = _sample_indices(total, n)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(_bgr_to_pil(frame))
    return frames


def _read_sequential(cap: cv2.VideoCapture, n: int) -> list[Image.Image]:
    """
    Read every frame sequentially (fallback when total frame count is unknown).
    Collects all frames then downsamples.
    """
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    if not all_frames:
        return []

    indices = _sample_indices(len(all_frames), n)
    return [_bgr_to_pil(all_frames[i]) for i in indices]


def _sample_indices(total: int, n: int) -> list[int]:
    """Return n evenly-spaced integer indices in [0, total)."""
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def _bgr_to_pil(frame: "np.ndarray") -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB Image."""
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


_VIDEO_MAGIC: dict[bytes, str] = {
    b"\x00\x00\x00\x18ftypmp4": ".mp4",
    b"\x00\x00\x00\x1cftypmp4": ".mp4",
    b"\x1aE\xdf\xa3": ".mkv",
    b"RIFF": ".avi",
    b"\x00\x00\x01\xba": ".mpeg",
}


def _guess_video_suffix(data: bytes) -> str:
    for magic, ext in _VIDEO_MAGIC.items():
        if data[: len(magic)] == magic:
            return ext
    return ".mp4"  # safe default
