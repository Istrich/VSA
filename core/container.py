"""Process-wide service container / dependency injection root.

Exists to guarantee that within a single Python process (Streamlit UI, CLI,
tests) we create exactly one ``ChromaVectorStore`` and one
``SQLiteMetadataDB``. ChromaDB's ``PersistentClient`` refuses a second
instance pointed at the same directory and the previous revision triggered
this every time a user switched between the Search and Index tabs
(BUG-N01/BUG-N22).

Consumers should either obtain services directly (``container.storage``,
``container.inference``, ``container.ollama``) or use the factory methods
``container.indexer()`` / ``container.search_engine()`` which wire
dependencies in one place.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import Settings, get_settings
from .db import ChromaVectorStore, SQLiteMetadataDB
from .model_downloader import ModelDownloader
from .models import ModelRegistry
from .vision import InferenceService

if TYPE_CHECKING:
    from .indexer import MediaIndexer, OllamaClient
    from .search import HybridSearchEngine


@dataclass
class Storage:
    """Bundle of long-lived storage services."""

    metadata_db: SQLiteMetadataDB
    vector_store: ChromaVectorStore


class ServiceContainer:
    """Holds one instance of each long-lived service for the process."""

    _instance: ServiceContainer | None = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.storage = Storage(
            metadata_db=SQLiteMetadataDB(settings=self.settings),
            vector_store=ChromaVectorStore(settings=self.settings),
        )
        self.storage.metadata_db.initialize()
        self.storage.vector_store.initialize()

        self.inference = InferenceService(settings=self.settings)
        self.model_registry = ModelRegistry(root=self.settings.insightface_home)
        self.model_downloader = ModelDownloader(settings=self.settings)
        self._ollama: OllamaClient | None = None

    @property
    def ollama(self) -> OllamaClient:
        """Lazy Ollama client (import-cycle safe)."""
        if self._ollama is None:
            from .indexer import OllamaClient  # local import to avoid cycle

            self._ollama = OllamaClient(settings=self.settings)
        return self._ollama

    def indexer(self) -> MediaIndexer:
        """Build a ``MediaIndexer`` bound to the shared services."""
        from .indexer import MediaIndexer

        return MediaIndexer(
            metadata_db=self.storage.metadata_db,
            vector_store=self.storage.vector_store,
            inference_service=self.inference,
            ollama_client=self.ollama,
            settings=self.settings,
        )

    def search_engine(self) -> HybridSearchEngine:
        """Build a ``HybridSearchEngine`` bound to the shared services."""
        from .search import HybridSearchEngine

        return HybridSearchEngine(
            metadata_db=self.storage.metadata_db,
            vector_store=self.storage.vector_store,
            inference_service=self.inference,
            settings=self.settings,
        )

    @classmethod
    def get(cls, settings: Settings | None = None) -> ServiceContainer:
        """Return the process-wide container, constructing it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget the current instance (used in tests)."""
        with cls._instance_lock:
            cls._instance = None
