"""Hybrid search engine for Vision Semantic Archive."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .db import ChromaVectorStore, SQLiteMetadataDB
from .models import SearchQuery, SearchWeights
from .vision import InferenceService

class HybridSearchEngine:
    """Performs hybrid retrieval and weighted reranking."""

    def __init__(
        self,
        metadata_db: SQLiteMetadataDB | None = None,
        vector_store: ChromaVectorStore | None = None,
        inference_service: InferenceService | None = None,
    ) -> None:
        self.metadata_db = metadata_db or SQLiteMetadataDB()
        self.vector_store = vector_store or ChromaVectorStore()
        self.inference_service = inference_service or InferenceService()

        self.metadata_db.initialize()
        self.vector_store.initialize()

    def search(
        self,
        query: SearchQuery | None = None,
        text_query: str = "",
        face_reference_path: str | None = None,
        top_k: int = 20,
        weights: SearchWeights | None = None,
    ) -> list[dict[str, Any]]:
        """Run CLIP+Face+FTS search and return ranked results."""
        if query is not None:
            text_query = query.text or ""
            face_reference_path = query.face_reference_path
            top_k = query.top_k
            if weights is None:
                weights = SearchWeights(**query.weights)
        scoring = weights or SearchWeights()
        candidates: dict[str, dict[str, Any]] = {}

        if text_query.strip():
            self._merge_clip_branch(candidates, text_query=text_query, top_k=top_k)
            self._merge_fts_branch(candidates, text_query=text_query, top_k=top_k)

        if face_reference_path:
            self._merge_face_branch(
                candidates, face_reference_path=face_reference_path, top_k=top_k
            )

        if not candidates:
            return []

        self._attach_metadata(candidates)
        self._compute_final_scores(candidates, scoring)

        ranked = sorted(
            candidates.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return ranked[:top_k]

    def _merge_clip_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        text_query: str,
        top_k: int,
    ) -> None:
        if self.vector_store.clip_collection is None:
            return

        text_embedding = self.inference_service.get_clip_text_embedding(text_query)
        result = self.vector_store.clip_collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for metadata, distance in zip(metadatas, distances):
            path = self._extract_path(metadata)
            if not path:
                continue
            clip_sim = 1.0 / (1.0 + float(distance))
            entry = candidates.setdefault(path, self._new_candidate(path))
            if clip_sim >= entry["clip_sim"]:
                entry["clip_sim"] = clip_sim
                timestamp = self._extract_timestamp_sec(metadata)
                if timestamp is not None:
                    entry["best_frame_timestamp_sec"] = timestamp

    def _merge_face_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        face_reference_path: str,
        top_k: int,
    ) -> None:
        if self.vector_store.face_collection is None:
            return

        faces = self.inference_service.get_faces(face_reference_path)
        if not faces:
            return
        reference_embedding = faces[0].get("embedding")
        if not isinstance(reference_embedding, list) or not reference_embedding:
            return

        result = self.vector_store.face_collection.query(
            query_embeddings=[reference_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for metadata, distance in zip(metadatas, distances):
            path = self._extract_path(metadata)
            if not path:
                continue
            face_sim = 1.0 / (1.0 + float(distance))
            entry = candidates.setdefault(path, self._new_candidate(path))
            if face_sim >= entry["face_sim"]:
                entry["face_sim"] = face_sim
                timestamp = self._extract_timestamp_sec(metadata)
                if timestamp is not None:
                    entry["best_frame_timestamp_sec"] = timestamp

    def _merge_fts_branch(
        self,
        candidates: dict[str, dict[str, Any]],
        text_query: str,
        top_k: int,
    ) -> None:
        rows = self.metadata_db.search_captions(text_query, limit=top_k)
        if not rows:
            return

        raw_scores = [float(row.get("fts_score", 0.0)) for row in rows]
        max_raw = max(raw_scores)
        min_raw = min(raw_scores)
        denom = (max_raw - min_raw) if max_raw != min_raw else 1.0

        for row in rows:
            path = row.get("path")
            if not isinstance(path, str):
                continue
            raw_score = float(row.get("fts_score", 0.0))
            # bm25 lower is better; invert to similarity-like [0..1]
            fts_sim = 1.0 - ((raw_score - min_raw) / denom)
            entry = candidates.setdefault(path, self._new_candidate(path))
            entry["fts_score"] = max(entry["fts_score"], fts_sim)

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
        clip_values = [float(entry["clip_sim"]) for entry in candidates.values()]
        face_values = [float(entry["face_sim"]) for entry in candidates.values()]

        clip_min, clip_max = min(clip_values), max(clip_values)
        face_min, face_max = min(face_values), max(face_values)

        for entry in candidates.values():
            entry["clip_sim"] = HybridSearchEngine._min_max_scale(
                float(entry["clip_sim"]), clip_min, clip_max
            )
            entry["face_sim"] = HybridSearchEngine._min_max_scale(
                float(entry["face_sim"]), face_min, face_max
            )
            entry["score"] = (
                (weights.clip * entry["clip_sim"])
                + (weights.face * entry["face_sim"])
                + (weights.fts * entry["fts_score"])
            )

    @staticmethod
    def _min_max_scale(value: float, min_value: float, max_value: float) -> float:
        if max_value == min_value:
            return 1.0 if value > 0.0 else 0.0
        return (value - min_value) / (max_value - min_value)

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
    ) -> list[dict[str, Any]]:
        """Convenience helper for UI file-uploader bytes."""
        if uploaded_bytes is None:
            return self.search(text_query=text_query, top_k=top_k, weights=weights)

        suffix = uploaded_suffix if uploaded_suffix.startswith(".") else f".{uploaded_suffix}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_bytes)
            tmp.flush()
            tmp_path = tmp.name
        try:
            return self.search(
                text_query=text_query,
                face_reference_path=tmp_path,
                top_k=top_k,
                weights=weights,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

