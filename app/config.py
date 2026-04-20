"""
Runtime configuration — all values from environment variables or .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Priority order:
      1. Real environment variables
      2. Values in .env file
      3. Defaults below (safe for local dev)
    """

    #  CLIP model (primary tree detector) 
    # Auto-downloaded from HuggingFace on first run — no API key needed.
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"

    # Prompts that mean "yes, tree is present". More prompts = better recall.
    TREE_POSITIVE_PROMPTS: str = (
        "a photo of a tree,"
        "a photo of trees,"
        "a photo of a forest,"
        "a photo of a palm tree,"
        "a photo of a plant or tree"
    )

    # Prompts that mean "no tree". Used to calibrate the softmax.
    TREE_NEGATIVE_PROMPTS: str = (
        "a photo with no trees,"
        "a photo of a building,"
        "a photo of a person,"
        "a photo of an indoor scene"
    )

    # If the best positive-prompt score exceeds this, tree_detected = True.
    CONFIDENCE_THRESHOLD: float = 0.40

    #  Upload limits 
    MAX_UPLOAD_BYTES: int = 10_485_760   # 10 MB
    ALLOWED_CONTENT_TYPES: str = "image/jpeg,image/png,image/webp"

    #  Server 
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #  Derived helpers 

    @property
    def positive_prompts(self) -> list[str]:
        return [p.strip() for p in self.TREE_POSITIVE_PROMPTS.split(",") if p.strip()]

    @property
    def negative_prompts(self) -> list[str]:
        return [p.strip() for p in self.TREE_NEGATIVE_PROMPTS.split(",") if p.strip()]

    @property
    def allowed_content_type_list(self) -> list[str]:
        return [c.strip() for c in self.ALLOWED_CONTENT_TYPES.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — safe to call from any module or FastAPI Depends()."""
    return Settings()
