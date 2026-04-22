"""Tests for async correctness of ``MediaIndexer`` (BUG-N08/N09/N10)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.indexer import MediaIndexer, OllamaClient
from core.models import MediaFile


class FakeInference:
    def __init__(self) -> None:
        self.face_calls = 0
        self.clip_batch_calls = 0

    def get_clip_embedding(self, image_path: Any) -> list[float]:
        return [0.1, 0.2, 0.3]

    def get_clip_embeddings(
        self, image_paths: list[Any], batch_size: int = 16
    ) -> list[list[float]]:
        self.clip_batch_calls += 1
        return [[0.1, 0.2, 0.3] for _ in image_paths]

    def get_faces(self, image_path: Any) -> list[dict[str, Any]]:
        self.face_calls += 1
        return []


class FakeOllama(OllamaClient):
    def __init__(self, *, caption_should_fail_for: set[int] | None = None) -> None:
        # Skip super().__init__ to avoid creating a real httpx client.
        self.caption_should_fail_for = caption_should_fail_for or set()
        self.caption_calls = 0
        self.summary_calls = 0

    async def caption_image(self, image_path: Any, model: Any = None, prompt: Any = None) -> str:
        self.caption_calls += 1
        if self.caption_calls - 1 in self.caption_should_fail_for:
            raise RuntimeError(f"forced failure on caption #{self.caption_calls}")
        return f"caption #{self.caption_calls}"

    async def summarize_video_captions(
        self, captions: list[str], model: Any = None
    ) -> str:
        self.summary_calls += 1
        return "summary"

    async def aclose(self) -> None:
        return None


class FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self.upserts.append({"ids": ids, "embeddings": embeddings, "metadatas": metadatas})

    def get(self, where: dict, include: list[str]) -> dict[str, Any]:
        ids: list[str] = []
        metas: list[dict] = []
        for entry in self.upserts:
            for i, m in zip(entry["ids"], entry["metadatas"], strict=True):
                if m.get("path") == where.get("path"):
                    ids.append(i)
                    metas.append(m)
        return {"ids": ids, "metadatas": metas}

    def update(self, ids: list[str], metadatas: list[dict]) -> None:
        self.updates.append({"ids": ids, "metadatas": metadatas})


class FakeVectorStore:
    def __init__(self) -> None:
        self._clip = FakeCollection()
        self._face = FakeCollection()
        self._settings = None

    def initialize(self) -> None:
        return None

    @property
    def clip_collection(self) -> FakeCollection:
        return self._clip

    @property
    def face_collection(self) -> FakeCollection:
        return self._face

    def rebind_media_path(self, old_path: str, new_path: str) -> int:
        total = 0
        for coll in (self._clip, self._face):
            fetched = coll.get(where={"path": old_path}, include=["metadatas"])
            ids = fetched.get("ids", [])
            metas = fetched.get("metadatas", [])
            if not ids:
                continue
            new_metas = [{**m, "path": new_path} for m in metas]
            coll.update(ids=ids, metadatas=new_metas)
            total += len(ids)
        return total


@pytest.fixture
def indexer(tmp_path: Path) -> MediaIndexer:
    from core.db import SQLiteMetadataDB

    metadata = SQLiteMetadataDB(db_path=tmp_path / "m.db")
    metadata.initialize()
    vector_store = FakeVectorStore()
    inference = FakeInference()
    ollama = FakeOllama()
    return MediaIndexer(
        metadata_db=metadata,
        vector_store=vector_store,  # type: ignore[arg-type]
        inference_service=inference,  # type: ignore[arg-type]
        ollama_client=ollama,  # type: ignore[arg-type]
    )


def _fake_image(path: Path, content: bytes = b"fake-image") -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class TestImageIndexing:
    def test_first_pass_indexes_image(self, indexer: MediaIndexer, tmp_path: Path) -> None:
        image = tmp_path / "a.jpg"
        _fake_image(image)
        stats = asyncio.run(indexer.index_directory(tmp_path))
        assert stats.indexed == 1
        assert stats.skipped == 0
        assert stats.failed == 0

    def test_second_pass_is_skipped(self, indexer: MediaIndexer, tmp_path: Path) -> None:
        image = tmp_path / "a.jpg"
        _fake_image(image)
        asyncio.run(indexer.index_directory(tmp_path))
        stats = asyncio.run(indexer.index_directory(tmp_path))
        assert stats.indexed == 0
        assert stats.skipped == 1

    def test_move_triggers_chroma_rebind(self, indexer: MediaIndexer, tmp_path: Path) -> None:
        src = tmp_path / "a.jpg"
        _fake_image(src, b"same-bytes")
        asyncio.run(indexer.index_directory(tmp_path))

        moved = tmp_path / "sub" / "a.jpg"
        moved.parent.mkdir()
        src.rename(moved)

        asyncio.run(indexer.index_directory(tmp_path))
        fake_store: FakeVectorStore = indexer.vector_store  # type: ignore[assignment]
        assert fake_store._clip.updates, "CLIP metadata was not rebound after move"
        new_path = fake_store._clip.updates[0]["metadatas"][0]["path"]
        assert new_path == str(moved.resolve())
