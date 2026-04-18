"""
Quick smoke-test / example usage for all three endpoints.

Usage:
  pip install httpx
  python test_api.py --image path/to/image.jpg
  python test_api.py --audio path/to/audio.flac
  python test_api.py --video path/to/video.mp4
  python test_api.py --image img.jpg --audio aud.flac --video vid.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # seconds — large video files need more time


def predict_image(path: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  /predict/image  ←  {Path(path).name}")
    print(f"{'='*55}")
    with open(path, "rb") as f:
        r = httpx.post(
            f"{BASE_URL}/predict/image?include_ensemble=true",
            files={"file": (Path(path).name, f, "image/jpeg")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    data = r.json()
    _print_result("Model V1 (EfficientNet-B0)", data["model_v1"])
    _print_result("Model V2 (EfficientNet-B3)", data["model_v2"])
    if data.get("ensemble"):
        _print_result("Ensemble", data["ensemble"])
    return data


def predict_audio(path: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  /predict/audio  ←  {Path(path).name}")
    print(f"{'='*55}")
    with open(path, "rb") as f:
        r = httpx.post(
            f"{BASE_URL}/predict/audio",
            files={"file": (Path(path).name, f, "audio/flac")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    data = r.json()
    _print_result("Audio Model (EfficientNet-B0)", data)
    return data


def predict_video(path: str, n_frames: int = 16) -> dict:
    print(f"\n{'='*55}")
    print(f"  /predict/video  ←  {Path(path).name}  ({n_frames} frames)")
    print(f"{'='*55}")
    with open(path, "rb") as f:
        r = httpx.post(
            f"{BASE_URL}/predict/video?n_frames={n_frames}&include_ensemble=true",
            files={"file": (Path(path).name, f, "video/mp4")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    data = r.json()
    _print_result("Model V1 (EfficientNet-B0)", data["model_v1"])
    _print_result("Model V2 (EfficientNet-B3)", data["model_v2"])
    audio = data["audio"]
    if audio["available"]:
        _print_result("Audio Model", audio)
    else:
        print("  Audio Model     : no audio track found")
    if data.get("ensemble"):
        _print_result("Ensemble", data["ensemble"])
    return data


def _print_result(label: str, result: dict) -> None:
    pred = result["prediction"].upper()
    conf = result.get("confidence", 0) * 100
    pr   = result.get("prob_real", 0) * 100
    pf   = result.get("prob_fake", 0) * 100
    print(
        f"  {label:<32}  {pred}  "
        f"({conf:.1f}% confident)  "
        f"[real={pr:.1f}%  fake={pf:.1f}%]"
    )


def health_check() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        ok = data.get("status") == "ok" and data.get("models_loaded")
        print(f"Health: {data}")
        return ok
    except Exception as exc:
        print(f"Health check failed: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection API smoke test")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--audio", help="Path to an audio file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--frames", type=int, default=16, help="Frames for video (default 16)")
    args = parser.parse_args()

    if not any([args.image, args.audio, args.video]):
        parser.print_help()
        sys.exit(1)

    if not health_check():
        print("\nServer is not healthy or models are not loaded yet. Aborting.")
        sys.exit(1)

    if args.image:
        predict_image(args.image)
    if args.audio:
        predict_audio(args.audio)
    if args.video:
        predict_video(args.video, n_frames=args.frames)
