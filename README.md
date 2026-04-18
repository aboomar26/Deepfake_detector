# Deepfake Detection API

A production-ready FastAPI backend for detecting AI-generated / deepfake
**images**, **videos**, and **audio** using three EfficientNet-based models
trained in the accompanying Jupyter notebooks.

---

## Project Structure

```
deepfake_api/
│
├── main.py                     # FastAPI app + lifespan (model loading)
├── config.py                   # All paths & hyperparameters
├── requirements.txt
├── Dockerfile
│
├── models/
│   └── architectures.py        # Exact model class definitions (NB1, NB2, NB3)
│
├── services/
│   ├── model_registry.py       # Singleton — loads all 3 models once at startup
│   ├── image_video_service.py  # run_v1() / run_v2() inference logic
│   └── audio_service.py        # predict_audio() inference logic
│
├── routers/
│   ├── image_router.py         # POST /predict/image
│   ├── video_router.py         # POST /predict/video
│   └── audio_router.py         # POST /predict/audio
│
└── utils/
    ├── image_utils.py          # PIL loading + transform pipelines
    ├── audio_utils.py          # load_audio() + audio_to_melspec() (NB2)
    ├── video_utils.py          # extract_frames() + extract_audio()
    └── ensemble.py             # Soft-voting ensemble helpers
```

---

## Model → Notebook Mapping

| Model | Notebook | Architecture | IMG_SIZE | Task |
|-------|----------|-------------|---------|------|
| Model V1 | Notebook 1 | EfficientNet-B0 | 128×128 | Image + Video |
| Model V2 | Notebook 3 (improved) | EfficientNet-B3 | 224×224 | Image + Video |
| Audio Model | Notebook 2 | EfficientNet-B0 (on Mel-spectrograms) | 128×128 | Audio |

---

## Quick Start

### 1. Install dependencies

```bash
# GPU (recommended)
pip install -r requirements.txt

# CPU only — also works, just replace torch line in requirements.txt with:
# torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

Also install **ffmpeg** (needed for audio extraction from video):

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 2. Place model checkpoints

Put your trained `.pth` files in a `checkpoints/` folder (or set env vars):

```
deepfake_api/
└── checkpoints/
    ├── best_deepfake_model.pth          ← Notebook 1 output
    ├── best_deepfake_model_v3.pth       ← Notebook 3 output
    └── best_audio_deepfake_model.pth    ← Notebook 2 output
```

Override paths with environment variables:

```bash
export MODEL_V1_PATH=/data/models/best_deepfake_model.pth
export MODEL_V2_PATH=/data/models/best_deepfake_model_v3.pth
export AUDIO_MODEL_PATH=/data/models/best_audio_deepfake_model.pth
```

### 3. Run the server

```bash
cd deepfake_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /predict/image`

Detect whether an image is real or AI-generated.

**Input:** multipart/form-data with field `file` (JPEG, PNG, BMP, WEBP)

**Query params:**
- `use_tta` (bool, default `false`) — Test-Time Augmentation on Model V2
- `include_ensemble` (bool, default `true`) — include weighted ensemble

**Example:**

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "file=@photo.jpg"
```

**Response:**

```json
{
  "model_v1": {
    "prediction": "fake",
    "confidence": 0.9231,
    "prob_real": 0.0769,
    "prob_fake": 0.9231
  },
  "model_v2": {
    "prediction": "fake",
    "confidence": 0.9654,
    "prob_real": 0.0346,
    "prob_fake": 0.9654
  },
  "ensemble": {
    "prediction": "fake",
    "confidence": 0.9486,
    "prob_real": 0.0514,
    "prob_fake": 0.9486,
    "weights_used": { "model_v1": 0.4, "model_v2": 0.6 }
  }
}
```

---

### `POST /predict/audio`

Detect whether an audio clip is real or AI-generated.

**Input:** multipart/form-data with field `file` (WAV, FLAC, MP3, OGG …)

**Example:**

```bash
curl -X POST http://localhost:8000/predict/audio \
  -F "file=@clip.flac"
```

**Response:**

```json
{
  "prediction": "fake",
  "confidence": 0.8812,
  "prob_real": 0.1188,
  "prob_fake": 0.8812
}
```

---

### `POST /predict/video`

Full multi-modal video analysis (visual frames + audio track).

**Input:** multipart/form-data with field `file` (MP4, AVI, MKV, MOV …)

**Query params:**
- `n_frames` (int 1–64, default `16`) — frames to sample from the video
- `use_tta` (bool, default `false`) — TTA on Model V2
- `include_ensemble` (bool, default `true`) — include combined decision

**Example:**

```bash
curl -X POST "http://localhost:8000/predict/video?n_frames=24" \
  -F "file=@video.mp4"
```

**Response:**

```json
{
  "model_v1": {
    "prediction": "fake",
    "confidence": 0.8743,
    "prob_real": 0.1257,
    "prob_fake": 0.8743,
    "frames_analysed": 16
  },
  "model_v2": {
    "prediction": "fake",
    "confidence": 0.9102,
    "prob_real": 0.0898,
    "prob_fake": 0.9102,
    "frames_analysed": 16
  },
  "audio": {
    "prediction": "fake",
    "confidence": 0.9311,
    "prob_real": 0.0689,
    "prob_fake": 0.9311,
    "available": true
  },
  "ensemble": {
    "prediction": "fake",
    "confidence": 0.9052,
    "prob_real": 0.0948,
    "prob_fake": 0.9052,
    "audio_included": true
  }
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "models_loaded": true}
```

---

## Docker

```bash
# Build
docker build -t deepfake-api .

# Run (mount your checkpoints directory)
docker run -p 8000:8000 \
  -v /absolute/path/to/checkpoints:/app/checkpoints \
  --gpus all \
  deepfake-api
```

For CPU-only:

```bash
docker run -p 8000:8000 \
  -v /absolute/path/to/checkpoints:/app/checkpoints \
  deepfake-api
```

---

## Ensemble Weights

Default weights (tunable in `utils/ensemble.py`):

| Endpoint | Model V1 | Model V2 | Audio |
|----------|---------|---------|-------|
| `/predict/image` | 0.40 | 0.60 | — |
| `/predict/video` | 0.25 | 0.35 | 0.40 |

Model V2 and audio get higher weights because:
- V2 uses EfficientNet-B3 (larger, more accurate than B0)
- Audio is an independent modality — deepfake audio is often the giveaway

---

## Notes

- **Models are loaded once at startup** — requests never reload weights.
- **Workers = 1** is strongly recommended when using GPU. Scale horizontally
  with multiple containers rather than multiple uvicorn workers.
- The server gracefully handles videos with **no audio track** — the audio
  field returns `"available": false` and the ensemble re-normalises weights.
- If a checkpoint file is missing the server still starts, but returns
  random-weight predictions. A warning is logged at startup.
