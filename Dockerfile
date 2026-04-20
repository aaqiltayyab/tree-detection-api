# ── Stage 1: download the CLIP model weights ─────────────────────────────────
# Separated into its own stage so Docker layer cache keeps the weights even
# when app code changes. A code-only change won't re-download 600 MB.
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir transformers huggingface_hub torch --index-url https://download.pytorch.org/whl/cpu

# HF_HOME controls where transformers caches model files inside the container.
ENV HF_HOME=/model_cache

# Download CLIP once. The cache is baked into this layer and reused by the
# final stage — zero download at container startup.
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP model cached successfully')"


# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-downloaded model cache from stage 1.
COPY --from=model-downloader /model_cache /model_cache

# Tell transformers/HF to use our baked-in cache — no internet needed at runtime.
ENV HF_HOME=/model_cache
ENV TRANSFORMERS_OFFLINE=1

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code last — code changes only invalidate this layer, not the model cache.
COPY . .

EXPOSE 8000

# 2 workers for production. Increase if you have more CPU cores.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
