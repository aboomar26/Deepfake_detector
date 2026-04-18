"""
Deepfake Detection API
======================
FastAPI backend for image, video, and audio deepfake detection.

Models:
  - Model 1 (image/video): EfficientNet-B0  — Notebook 1  (best_deepfake_model.pth)
  - Model 2 (image/video): EfficientNet-B3  — Notebook 3  (best_deepfake_model_v3.pth)
  - Audio model:           EfficientNet-B0  — Notebook 2  (best_audio_deepfake_model.pth)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import image_router, video_router, audio_router
from services.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; release on shutdown."""
    logger.info("Loading all deepfake detection models...")
    registry = ModelRegistry.get_instance()
    registry.load_all()
    logger.info("All models loaded and ready.")
    yield
    logger.info("Shutting down — releasing models.")
    registry.unload_all()


app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "Detect AI-generated or deepfake media (image, video, audio) "
        "using EfficientNet-based models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router.router, prefix="/predict", tags=["Image"])
app.include_router(video_router.router, prefix="/predict", tags=["Video"])
app.include_router(audio_router.router, prefix="/predict", tags=["Audio"])


@app.get("/health", tags=["Health"])
async def health():
    """Quick liveness check."""
    registry = ModelRegistry.get_instance()
    return {
        "status": "ok",
        "models_loaded": registry.are_loaded(),
    }
