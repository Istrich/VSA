"""Hybrid search engine for Vision Semantic Archive.

Changes vs. previous revision:

- Public API accepts a pydantic ``SearchQuery`` and returns typed
  ``SearchResult`` objects (TYPES-N01, nestled into ``.cursorrules``).
- Similarity normalization happens **per-branch** before the merge, not
  globally after, so a FTS-only hit no longer gets a bogus normalized
  clip/face score (BUG-N17/N18).
- ``best_frame_timestamp_sec`` is picked by the highest similarity in its
  branch, not by arrival order (BUG-N19).
- ``search_with_uploaded_face`` uses ``TemporaryDirectory`` so the uploaded
  bytes are always cleaned up even if search raises (BUG-N21).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .db import ChromaVectorStore, SQLiteMetadataDB
from .exceptions import SearchError
from .models import SearchQuery, SearchResult, SearchWeights
from .vision import InferenceService

LOGGER = logging.getLogger(__name__)


class HybridSearchEngine:
    """Performs hybrid retrieval and weighted reranking."""

    def __init__(
        self,
        metadata_db: SQLiteMetadataDB | None = None,
        vector_store: ChromaVectorStore | None = None,
        inference_service: InferenceService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.metadata_db = metadata_db or SQLiteMetadataDB(settings=self._settings)
        self.vector_store = vector_store or ChromaVectorStore(settings=self._settings)
        self.inference_service = inference_service or InferenceService(settings=self._settings)
        self.metadata_db.initialize()
        self.vector_store.initialize()

    def search(
        self,
        query: SearchQuery | None = None,
        *,
        text_query: str = "",
        face_reference_path: str | None = None,
        top_k: int = 20,
        weights: SearchWeights | None = None,
    ) -> list[SearchResult]:
        """Run CLIP+Face+FTS search and return ranked ``SearchResult`` list.

        Accepts either a pydantic ``SearchQuery`` or flat kwargs. The flat
        kwargs path is kept for the Streamlit callbacks.
        """
        resolved_query = query or SearchQuery(
            text=text_query or None,
            face_reference_path=face_reference_path,
            top_k=top_k,
            weights=weights or SearchWeights(),
        )
        candidates: dict[str, dict[str, Any]] = {}

        text = (resolved_query.text or "").strip()
        if text:
            self._merge_clip_branch(candidates, text_query=text, top_k=resolved_query.top_k)
            self._merge_fts_branch(candidates, text_query=text, top_k=resolved_query.top_k)

        if resolved_query.face_reference_path:
            self._merge_face_branch(
                candidates,
                face_reference_path=resolved_query.face_reference_path,
                top_k=resolved_query.top_k,
            )

        if not candidates:
            return []

        self._attach_metadata(candidates)
        self._compute_final_scores(candidates, resolved_query.weights)

        ranked = sorted(
            candidates.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return [self._to_search_result(item) for item in ranked[: resolved_query.top_k]]

    def _merge_clip_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        text_query: str,
        top_k: int,
    ) -> None:
        try:
            text_embedding = self.inference_service.get_clip_text_embedding(text_query)
        except Exception:
            LOGGER.exception("CLIP text embedding failed for query: %s", text_query)
            return

        try:
            result = self.vector_store.clip_collection.query(
                query_embeddings=[text_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
        except Exception:
            LOGGER.exception("Chroma clip query failed.")
            return

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not metadatas:
            return

        raw: list[tuple[str, float, float | None]] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            path = self._extract_path(metadata)
            if not path:
                continue
            sim_raw = 1.0 / (1.0 + float(distance))
            raw.append((path, sim_raw, self._extract_timestamp_sec(metadata)))
        if not raw:
            return

        normalized = self._normalize_pairs([r[1] for r in raw])
        best_by_path: dict[str, tuple[float, float | None]] = {}
        for (path, _raw_sim, ts), norm_sim in zip(raw, normalized, strict=True):
            current = best_by_path.get(path)
            if current is None or norm_sim > current[0]:
                best_by_path[path] = (norm_sim, ts)

        for path, (norm_sim, ts) in best_by_path.items():
            entry = candidates.setdefault(path, self._new_candidate(path))
            if norm_sim > entry["clip_sim"]:
                entry["clip_sim"] = norm_sim
                if ts is not None:
                    entry["best_frame_timestamp_sec"] = ts

    def _merge_face_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        face_reference_path: str,
        top_k: int,
    ) -> None:
        try:
            faces = self.inference_service.get_faces(face_reference_path)
        except Exception:
            LOGGER.exception("Face reference embedding failed: %s", face_reference_path)
            return
        if not faces:
            return
        reference_embedding = faces[0].get("embedding")
        if not isinstance(reference_embedding, list) or not reference_embedding:
            return

        try:
            result = self.vector_store.face_collection.query(
                query_embeddings=[reference_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
        except Exception:
            LOGGER.exception("Chroma face query failed.")
            return

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not metadatas:
            return

        raw: list[tuple[str, float, float | None]] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            path = self._extract_path(metadata)
            if not path:
                continue
            sim_raw = 1.0 / (1.0 + float(distance))
            raw.append((path, sim_raw, self._extract_timestamp_sec(metadata)))
        if not raw:
            return

        normalized = self._normalize_pairs([r[1] for r in raw])
        best_by_path: dict[str, tuple[float, float | None]] = {}
        for (path, _raw_sim, ts), norm_sim in zip(raw, normalized, strict=True):
            current = best_by_path.get(path)
            if current is None or norm_sim > current[0]:
                best_by_path[path] = (norm_sim, ts)

        for path, (norm_sim, ts) in best_by_path.items():
            entry = candidates.setdefault(path, self._new_candidate(path))
            if norm_sim > entry["face_sim"]:
                entry["face_sim"] = norm_sim
                if ts is not None:
                    entry["best_frame_timestamp_sec"] = ts

    def _merge_fts_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        text_query: str,
        top_k: int,
    ) -> None:
        rows = self.metadata_db.search_captions(text_query, limit=top_k)
        if not rows:
            return

        # bm25 returns scores where smaller is "better"; convert to positive
        # "relevance" via negation before min-max so that larger -> closer to
        # 1.0 after normalization.
        relevance = [-float(row.get("fts_score", 0.0)) for row in rows]
        normalized = self._normalize_pairs(relevance)
        for row, norm_score in zip(rows, normalized, strict=True):
            path = row.get("path")
            if not isinstance(path, str):
                continue
            entry = candidates.setdefault(path, self._new_candidate(path))
            entry["fts_score"] = max(entry["fts_score"], norm_score)

    def _attach_metadata(self, candidates: dict[str, dict[str, Any]]) -> None:
        paths = list(candidates.keys())
        rows = self.metadata_db.get_media_by_paths(paths)
        by_path = {row["path"]: row for row in rows if isinstance(row.get("path"), str)}

        for path, entry in candidates.items():
            row = by_path.get(path)
            if row is None:
                continue
            metadata_json = row.get("metadata_json", "{}")
            try:
                parsed_metadata = (
                    json.loads(metadata_json)
                    if isinstance(metadata_json, str)
                    else dict(metadata_json)
                )
            except (TypeError, json.JSONDecodeError):
                parsed_metadata = {}

            entry["id"] = row.get("id")
            entry["caption"] = row.get("caption")
            entry["created_at"] = row.get("created_at")
            entry["metadata"] = parsed_metadata
            best_timestamp = entry.get("best_frame_timestamp_sec")
            if isinstance(best_timestamp, float):
                entry["metadata"]["best_frame_timestamp_sec"] = best_timestamp

    @staticmethod
    def _compute_final_scores(
        candidates: dict[str, dict[str, Any]],
        weights: SearchWeights,
    ) -> None:
        """Compute weighted score from already per-branch-normalized similarities."""
        for entry in candidates.values():
            clip_sim = float(entry["clip_sim"])
            face_sim = float(entry["face_sim"])
            fts_score = float(entry["fts_score"])
            entry["score"] = (
                weights.clip * clip_sim + weights.face * face_sim + weights.fts * fts_score
            )

    @staticmethod
    def _normalize_pairs(values: list[float]) -> list[float]:
        """Min-max normalise a list to [0..1].

        Degenerate input (empty / single value / all equal) returns the list
        unchanged at 1.0 for non-zero and 0.0 for zero, matching the
        expectation that a lone candidate should not collapse to score=0.
        """
        if not values:
            return []
        if len(values) == 1:
            return [1.0 if values[0] > 0.0 else 0.0]
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [1.0 if v > 0.0 else 0.0 for v in values]
        span = v_max - v_min
        return [(v - v_min) / span for v in values]

    @staticmethod
    def _new_candidate(path: str) -> dict[str, Any]:
        return {
            "path": path,
            "score": 0.0,
            "clip_sim": 0.0,
            "face_sim": 0.0,
            "fts_score": 0.0,
            "id": None,
            "caption": None,
            "created_at": None,
            "metadata": {},
            "best_frame_timestamp_sec": None,
        }

    @staticmethod
    def _to_search_result(entry: dict[str, Any]) -> SearchResult:
        return SearchResult(
            path=str(entry.get("path", "")),
            score=float(entry.get("score", 0.0)),
            clip_sim=float(entry.get("clip_sim", 0.0)),
            face_sim=float(entry.get("face_sim", 0.0)),
            fts_score=float(entry.get("fts_score", 0.0)),
            id=entry.get("id"),
            caption=entry.get("caption"),
            created_at=entry.get("created_at"),
            metadata=entry.get("metadata") or {},
            best_frame_timestamp_sec=entry.get("best_frame_timestamp_sec"),
        )

    @staticmethod
    def _extract_path(metadata: Any) -> str | None:
        if not isinstance(metadata, dict):
            return None
        path = metadata.get("path")
        if isinstance(path, str) and path.strip():
            return path
        return None

    @staticmethod
    def _extract_timestamp_sec(metadata: Any) -> float | None:
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("frame_timestamp_sec")
        if isinstance(value, (float, int)):
            return float(value)
        return None

    def search_with_uploaded_face(
        self,
        text_query: str,
        uploaded_bytes: bytes | None,
        uploaded_suffix: str = ".jpg",
        top_k: int = 20,
        weights: SearchWeights | None = None,
    ) -> list[SearchResult]:
        """Convenience helper for UI file-uploader bytes (no temp-file leak)."""
        if uploaded_bytes is None:
            return self.search(text_query=text_query, top_k=top_k, weights=weights)

        suffix = uploaded_suffix if uploaded_suffix.startswith(".") else f".{uploaded_suffix}"
        with tempfile.TemporaryDirectory(prefix="vsa_face_ref_") as tmp_dir:
            tmp_path = Path(tmp_dir) / f"ref{suffix}"
            tmp_path.write_bytes(uploaded_bytes)
            try:
                return self.search(
                    text_query=text_query,
                    face_reference_path=str(tmp_path),
                    top_k=top_k,
                    weights=weights,
                )
            except Exception as exc:
                raise SearchError("Hybrid search with uploaded face failed.") from exc
