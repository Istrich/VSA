"""Centralised runtime configuration for Vision Semantic Archive.

Reads values from environment variables and/or a local ``.env`` file via
``pydantic-settings``. Every hard-coded path/URL in the previous revision was
moved here so deployment targets (dev laptop, Docker, RTX 3090 box) can tune
behaviour without touching source code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VSA runtime settings. All fields are overridable via env or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VSA_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    data_dir: Path = Field(default=Path("./data"))
    models_dir: Path = Field(default=Path("./models"))
    insightface_home: Path = Field(
        default=Path("./models/insightface_home"),
        description=(
            "Root passed to insightface.FaceAnalysis(root=...). FaceAnalysis "
            "resolves weights at <root>/models/<name>/. ModelDownloader "
            "extracts the buffalo_l pack into that location."
        ),
    )

    sqlite_filename: str = Field(default="metadata.db")
    chroma_subdir: str = Field(default="chroma")
    chroma_clip_collection: str = Field(default="embeddings_clip")
    chroma_faces_collection: str = Field(default="embeddings_faces")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_num_parallel: int = Field(default=1, ge=1, le=64)
    ollama_timeout_sec: float = Field(default=120.0, gt=0.0)
    ollama_max_retries: int = Field(default=3, ge=1, le=10)
    ollama_caption_model: str = Field(default="moondream2")
    ollama_summary_model: str = Field(default="llama3")

    clip_model_name: str = Field(default="ViT-L-14")
    clip_pretrained: str = Field(
        default="hf-hub:openai/clip-vit-large-patch14",
        description=(
            "open_clip `pretrained` argument. Accepts either a hf-hub spec, "
            "a built-in tag (e.g. 'openai'), or a local checkpoint path."
        ),
    )
    face_model_name: str = Field(default="buffalo_l")

    allow_cpu: bool = Field(default=True)
    cuda_runtime_pin: str | None = Field(
        default=None,
        description=(
            "Optional CUDA runtime version prefix (e.g. '12.1') to assert at "
            "startup. None disables the check; useful for dev laptops."
        ),
    )

    log_level: str = Field(default="INFO")
    max_upload_size_mb: int = Field(default=10, ge=1, le=1024)

    @field_validator("data_dir", "models_dir", "insightface_home", mode="before")
    @classmethod
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(str(value)).expanduser()

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / self.chroma_subdir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide cached Settings instance."""
    return Settings()  # type: ignore[call-arg]
