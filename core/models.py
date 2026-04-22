"""Pydantic data models and model-asset registry for Vision Semantic Archive.

All cross-module data now travels through pydantic models per `.cursorrules`:

- ``MediaFile`` — SQLite row projection for media entities
- ``Face`` — detected face descriptor
- ``SearchQuery``/``SearchWeights``/``SearchResult`` — public search API
- ``IndexingStats`` — return type of ``MediaIndexer.index_directory``
- ``ModelSpec``/``ModelRegistry`` — on-disk asset registry (service, not data)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    bbox: list[float] = Field(description="Face bounding box in XYXY format: [x1, y1, x2, y2].")
    embedding_id: str


class SearchWeights(BaseModel):
    """Normalized hybrid scoring weights."""

    clip: float = Field(default=0.5, ge=0.0, le=1.0)
    face: float = Field(default=0.4, ge=0.0, le=1.0)
    fts: float = Field(default=0.1, ge=0.0, le=1.0)


class SearchQuery(BaseModel):
    """Hybrid search query payload."""

    text: str | None = None
    face_reference_path: str | None = None
    top_k: int = Field(default=20, ge=1, le=500)
    weights: SearchWeights = Field(default_factory=SearchWeights)


class SearchResult(BaseModel):
    """Search API result payload returned by ``HybridSearchEngine.search``."""

    path: str
    score: float
    clip_sim: float = 0.0
    face_sim: float = 0.0
    fts_score: float = 0.0
    id: str | None = None
    caption: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    best_frame_timestamp_sec: float | None = None


class IndexingStats(BaseModel):
    """Aggregate counters produced by ``MediaIndexer.index_directory``."""

    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    total_candidates: int = 0


class ModelSpec(BaseModel):
    """Single model artifact expected on disk."""

    name: str
    relative_path: str = Field(
        description=(
            "File whose presence declares the pack ready. Path is relative to "
            "the registry root (defaults to Settings.models_dir)."
        )
    )
    download_url: str
    sha256: str | None = Field(
        default=None,
        description="Optional digest of the downloaded artifact (zip or file).",
    )
    archive_sha256: str | None = Field(
        default=None,
        description=(
            "Optional sha256 of the raw downloaded archive *before* extraction. "
            "Populated for zipped packs such as buffalo_l."
        ),
    )
    required: bool = True

    @field_validator("relative_path")
    @classmethod
    def _no_absolute_paths(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("relative_path must be relative to the registry root")
        return value


class ModelRegistry:
    """Checks whether required model files are present.

    The registry now reflects what the code actually loads:

    - CLIP weights are fetched by ``open_clip`` itself (from the HuggingFace
      hub via the ``hf-hub:...`` spec). We do not ship a local safetensors
      copy any more — the orphaned asset from the previous revision was the
      source of BUG-N06/N25.
    - InsightFace expects ``<root>/models/<name>/...``. The ``buffalo_l``
      pack is downloaded as a zip and extracted into
      ``<insightface_home>/models/buffalo_l/`` which matches
      ``FaceAnalysis(name="buffalo_l", root=insightface_home)``.
    """

    def __init__(self, root: str | Path = "./models/insightface_home") -> None:
        self.root = Path(root)
        self.required_models: tuple[ModelSpec, ...] = (
            ModelSpec(
                name="insightface_buffalo_l",
                relative_path="models/buffalo_l/w600k_r50.onnx",
                download_url=(
                    "https://github.com/deepinsight/insightface/releases/"
                    "download/v0.7/buffalo_l.zip"
                ),
                archive_sha256=None,
                sha256=None,
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
        """Return True when all required model files are present on disk."""
        return all(status == "FOUND" for status in self.get_model_status().values())

    def get_specs(self) -> tuple[ModelSpec, ...]:
        """Return immutable model specs registry."""
        return self.required_models
