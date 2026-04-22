"""Data models and model asset status for Vision Semantic Archive."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Processing lifecycle status for media files."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class MediaFile(BaseModel):
    """Metadata entity for indexed media."""

    id: str
    path: str
    file_hash: str = Field(alias="hash")
    caption: str | None = None
    created_at: datetime
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.pending


class Face(BaseModel):
    """Detected face descriptor for media/frame."""

    id: str
    media_id: str
    bbox: list[float] = Field(
        description="Face bounding box in XYXY format: [x1, y1, x2, y2]."
    )
    embedding_id: str


class SearchQuery(BaseModel):
    """Hybrid search query payload."""

    text: str | None = None
    face_reference_path: str | None = None
    top_k: int = 20
    weights: dict[str, float] = Field(
        default_factory=lambda: {"clip": 0.5, "face": 0.4, "fts": 0.1}
    )


class SearchWeights(BaseModel):
    """Normalized hybrid scoring weights."""

    clip: float = 0.5
    face: float = 0.4
    fts: float = 0.1


class SearchResult(BaseModel):
    """Search API result payload."""

    path: str
    score: float
    clip_sim: float
    face_sim: float
    fts_score: float
    id: str | None = None
    caption: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    best_frame_timestamp_sec: float | None = None


class ModelSpec(BaseModel):
    """Single model artifact expected on disk."""

    name: str
    relative_path: str
    download_url: str
    sha256: str | None = None


class ModelRegistry:
    """Checks whether required model files are present."""

    def __init__(self, root: str | Path = "./models") -> None:
        self.root = Path(root)
        self.required_models: tuple[ModelSpec, ...] = (
            ModelSpec(
                name="clip_vit_l_14",
                relative_path="vision/open_clip_vit_l_14/model.safetensors",
                download_url=(
                    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"
                ),
            ),
            ModelSpec(
                name="insightface_buffalo_l",
                relative_path="faces/buffalo_l/w600k_r50.onnx",
                download_url=(
                    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
                ),
            ),
        )

    def get_model_status(self) -> dict[str, str]:
        """Return FOUND/MISSING status for each required model."""
        result: dict[str, str] = {}
        for spec in self.required_models:
            model_path = self.root / spec.relative_path
            result[spec.name] = "FOUND" if model_path.exists() else "MISSING"
        return result

    def is_ready(self) -> bool:
        """Return True when all required models are found on disk."""
        return all(status == "FOUND" for status in self.get_model_status().values())

    def get_specs(self) -> tuple[ModelSpec, ...]:
        """Return immutable model specs registry."""
        return self.required_models

