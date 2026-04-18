# ── Base image ──────────────────────────────────────────────────────────────
# Use the official PyTorch image with CUDA support.
# For CPU-only deployment replace with: python:3.11-slim
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── System dependencies ──────────────────────────────────────────────────────
# ffmpeg  → audio extraction from video
# libgl1  → OpenCV headless needs this
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Model checkpoints ────────────────────────────────────────────────────────
# Mount your checkpoint directory at runtime or copy it in here:
#   docker run -v /host/checkpoints:/app/checkpoints deepfake-api
RUN mkdir -p checkpoints

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# NOTE: keep workers=1 — each worker loads all three models into GPU memory.
#       Scale horizontally with multiple containers instead.
