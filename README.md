# Tree-AI Detection API

Upload any photo → get back whether it contains a tree.  
Powered by **OpenAI CLIP** (zero-shot, free, no API key).

---

## How it works

CLIP understands natural language. Instead of training a model to recognize a "tree" class, we ask it:

> _"Does this image look more like 'a photo of a tree' or 'a photo with no trees'?"_

This means it correctly detects trees even when people, buildings, or vehicles are also in the frame — no fine-tuning, no labelled dataset.

---

## Quick start

```bash
# 1. Clone
git clone <repo-url> && cd tree-ai

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
copy .env.example .env           # Linux/Mac: cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# 5. Run
uvicorn app.main:app --reload
```

First startup downloads the CLIP model (~600 MB) from HuggingFace automatically.  
Subsequent starts use the local cache — no internet needed.

- **API:** `http://localhost:8000/api/v1/detect`
- **Docs:** `http://localhost:8000/docs`

---

## API

### `POST /api/v1/detect`

Upload a JPG, PNG, or WEBP image.

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@photo.jpg"
```

**Tree found:**

```json
{
  "tree_detected": true,
  "confidence": 0.823,
  "label": "a photo of a tree",
  "all_detections": [
    "a photo of a tree(0.82)",
    "a photo of trees(0.11)",
    "a photo with no trees(0.04)"
  ]
}
```

**No tree:**

```json
{
  "tree_detected": false,
  "confidence": 0.0,
  "message": "No tree found. Please upload a photo with a tree.",
  "all_detections": [
    "a photo of a building(0.61)",
    "a photo with no trees(0.28)"
  ]
}
```

**Error codes:**

| Code | Meaning                   |
| ---- | ------------------------- |
| 400  | Empty or unreadable image |
| 413  | File exceeds 10 MB limit  |
| 422  | Unsupported file type     |
| 500  | Inference error           |

---

## Configuration (`.env`)

| Variable                | Default                        | Description                        |
| ----------------------- | ------------------------------ | ---------------------------------- |
| `CLIP_MODEL`            | `openai/clip-vit-base-patch32` | HuggingFace model ID               |
| `TREE_POSITIVE_PROMPTS` | _(5 prompts)_                  | Comma-separated "yes tree" prompts |
| `TREE_NEGATIVE_PROMPTS` | _(4 prompts)_                  | Comma-separated "no tree" prompts  |
| `CONFIDENCE_THRESHOLD`  | `0.40`                         | Minimum score to accept as tree    |
| `MAX_UPLOAD_BYTES`      | `10485760`                     | Max upload size (10 MB)            |
| `APP_ENV`               | `development`                  | `production` disables `/docs`      |
| `LOG_LEVEL`             | `INFO`                         | `DEBUG` / `INFO` / `WARNING`       |

**Tuning tip:** If you get false positives, raise `CONFIDENCE_THRESHOLD` to `0.50`.  
If real trees are missed, lower it to `0.30` or add more positive prompts.

---

## Project structure

```
tree-ai/
├── app/
│   ├── config.py    ← all settings from env vars
│   ├── schemas.py   ← Pydantic request/response models
│   ├── model.py     ← CLIP loading + inference (no HTTP here)
│   ├── router.py    ← FastAPI routes (no ML here)
│   └── main.py      ← app factory + lifespan
├── .env.example     ← copy to .env
├── .gitignore
├── Dockerfile
└── requirements.txt
```

**To add a new AI feature** (e.g. species classification):

1. Add inference logic in `model.py`
2. Add a response schema in `schemas.py`
3. Add a new route in `router.py`
   Nothing else needs to change.

---

## Docker

```bash
# Build (downloads CLIP during build — not at runtime)
docker build -t tree-ai .

# Run
docker run --env-file .env -p 8000:8000 tree-ai
```

The Dockerfile uses a **multi-stage build**: CLIP weights are baked into the image at build time, so containers start instantly with no internet dependency.

---

## Cloud deployment

All platforms build from `Dockerfile` automatically. Set env vars from `.env.example` in the platform dashboard.

| Platform                    | Notes                                                                |
| --------------------------- | -------------------------------------------------------------------- |
| **Railway**                 | Connect repo → set env vars → deploy. Uses Dockerfile automatically. |
| **Render**                  | New Web Service → Docker → set env vars.                             |
| **Fly.io**                  | `fly launch` → `fly deploy`.                                         |
| **AWS ECS / GCP Cloud Run** | Push Docker image to ECR/GCR, deploy as a service.                   |

**Persistent model cache (optional):** If you don't want to bake the model into the image, mount a persistent volume and set `HF_HOME=/your/volume/path` — the model downloads once and survives redeploys.
