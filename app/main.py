"""
HOW TO RUN
──────────
  # 1. Install dependencies (once)
  pip install -r requirements.txt
 
  # 2. Create your .env file (once)
  cp .env.example .env          # Windows: copy .env.example .env

  # 3. Start the server
  uvicorn app.main:app --reload                    # dev  → http://localhost:8000
  uvicorn app.main:app --host 0.0.0.0 --port 8000  # prod (no reload)

  # 4. Open API docs
  http://localhost:8000/docs

  # 5. Test the endpoint
  curl -X POST http://localhost:8000/api/v1/detect -F "file=@photo.jpg"

  NOTE: First startup downloads the CLIP model (~600 MB) from HuggingFace.
        Subsequent starts use the local cache — no internet needed.
──────────
Application entry point.

Responsibilities:
  - Create the FastAPI app.
  - Register the startup/shutdown lifespan (model loading).
  - Configure structured logging.
  - Mount the router.
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.model import load_model
from app.router import router


#  Logging 

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


#  Lifespan 

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the CLIP model once at startup; release on shutdown."""
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)

    logger = logging.getLogger(__name__)
    logger.info("Starting tree-ai API (env=%s)", settings.APP_ENV)

    load_model(settings)   # blocks until model is downloaded + on device

    yield

    logger.info("Shutting down tree-ai API")


#  App factory 

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Tree-AI Detection API",
        description=(
            "Upload an image and find out whether it contains a tree. "
            "Powered by OpenAI CLIP zero-shot classification."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


#  Dev runner 

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
