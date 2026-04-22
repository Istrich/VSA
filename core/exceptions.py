"""Typed VSA exceptions used across modules for explicit error handling."""

from __future__ import annotations


class VSAError(Exception):
    """Base class for all VSA domain errors."""


class ConfigError(VSAError):
    """Invalid or missing configuration."""


class StorageError(VSAError):
    """SQLite / ChromaDB storage failures."""


class ModelAssetError(VSAError):
    """Missing or corrupted model artifacts on disk."""


class InferenceError(VSAError):
    """CLIP / InsightFace inference failures."""


class OllamaError(VSAError):
    """Remote Ollama API errors (transport, schema, retries exhausted)."""


class IngestError(VSAError):
    """Indexing pipeline failed for a specific media file."""


class SearchError(VSAError):
    """Search pipeline failures."""
