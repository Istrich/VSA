"""Media indexing pipeline with async Ollama integration."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx

from .db import ChromaVectorStore, SQLiteMetadataDB
from .models import MediaFile
from .vision import InferenceService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MOONDREAM_PROMPT = (
    "Describe this image concisely, focusing on main subjects, their actions, "
    "and the environment. Mention specific brands or landmarks if visible."
)


@dataclass(frozen=True)
class ExtractedKeyframe:
    """Keyframe file plus its source timestamp in seconds."""

    path: Path
    timestamp_sec: float


class OllamaClient:
    """Async client for local Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        num_parallel: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)
        env_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "1").strip()
        if num_parallel is not None:
            resolved_parallel = num_parallel
        else:
            try:
                resolved_parallel = int(env_parallel or "1")
            except ValueError:
                resolved_parallel = 1
        self.num_parallel = max(1, resolved_parallel)
        self._semaphore = asyncio.Semaphore(self.num_parallel)

    async def _post_generate(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/api/generate"
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON response.") from exc

        text = data.get("response", "")
        if not isinstance(text, str):
            raise RuntimeError("Ollama response payload does not contain text field.")
        return text.strip()

    async def caption_image(
        self,
        image_path: str | Path,
        model: str = "moondream2",
        prompt: str = MOONDREAM_PROMPT,
    ) -> str:
        """Generate concise image caption via VLM."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for captioning: {path}")

        image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }
        return await self._post_generate(payload)

    async def summarize_video_captions(
        self,
        captions: list[str],
        model: str = "llama3",
    ) -> str:
        """Aggregate frame captions into one video-level summary."""
        if not captions:
            return ""

        bullet_list = "\n".join(f"- {caption}" for caption in captions if caption.strip())
        prompt = (
            "You are summarizing keyframes from one video. "
            "Write one concise paragraph (max 80 words) that merges repeated details, "
            "mentions main subjects, actions and setting.\n\n"
            f"Frame descriptions:\n{bullet_list}"
        )
        payload = {"model": model, "prompt": prompt, "stream": False}
        return await self._post_generate(payload)


class MediaIndexer:
    """Coordinates media scanning and indexing into metadata/vector stores."""

    def __init__(
        self,
        metadata_db: SQLiteMetadataDB | None = None,
        vector_store: ChromaVectorStore | None = None,
        inference_service: InferenceService | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.metadata_db = metadata_db or SQLiteMetadataDB()
        self.vector_store = vector_store or ChromaVectorStore()
        self.inference_service = inference_service or InferenceService()
        self.ollama_client = ollama_client or OllamaClient()

        self.metadata_db.initialize()
        self.vector_store.initialize()

    async def index_directory(
        self,
        root_directory: str | Path,
        keyframe_interval_sec: int = 3,
        scene_delta_threshold: float = 15.0,
    ) -> dict[str, int]:
        """Recursively index all supported media files under root."""
        root = Path(root_directory)
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")

        counters = {"indexed": 0, "skipped": 0, "failed": 0}
        for media_path in root.rglob("*"):
            if not media_path.is_file():
                continue

            suffix = media_path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                ok = await self._index_image_file(media_path)
            elif suffix in VIDEO_EXTENSIONS:
                ok = await self._index_video_file(
                    media_path,
                    keyframe_interval_sec=keyframe_interval_sec,
                    scene_delta_threshold=scene_delta_threshold,
                )
            else:
                continue

            if ok is True:
                counters["indexed"] += 1
            elif ok is False:
                counters["failed"] += 1
            else:
                counters["skipped"] += 1
        return counters

    async def _index_image_file(self, image_path: Path) -> bool | None:
        file_hash = self._hash_file(image_path)
        if self.metadata_db.media_exists_by_hash(file_hash):
            return None

        try:
            clip_vector = self.inference_service.get_clip_embedding(image_path)
            faces = self.inference_service.get_faces(image_path)
            caption = await self.ollama_client.caption_image(image_path)
        except Exception:
            return False

        media_id = self._build_media_id(image_path)
        media = MediaFile(
            id=media_id,
            path=str(image_path.resolve()),
            hash=file_hash,
            caption=caption,
            created_at=datetime.now(timezone.utc),
            metadata_json={"type": "image", "face_count": len(faces)},
        )
        self.metadata_db.upsert_media(media)
        self._upsert_clip_embedding(media_id, clip_vector, media.path)
        self._upsert_face_embeddings(media_id, faces, media.path)
        return True

    async def _index_video_file(
        self,
        video_path: Path,
        keyframe_interval_sec: int,
        scene_delta_threshold: float,
    ) -> bool | None:
        file_hash = self._hash_file(video_path)
        if self.metadata_db.media_exists_by_hash(file_hash):
            return None

        video_media_id = self._build_media_id(video_path, prefix="video")
        try:
            with tempfile.TemporaryDirectory(prefix="vsa_keyframes_") as tmp_dir:
                extracted_keyframes = self._extract_keyframes(
                    video_path=video_path,
                    output_dir=Path(tmp_dir),
                    interval_sec=keyframe_interval_sec,
                    scene_delta_threshold=scene_delta_threshold,
                )
                if not extracted_keyframes:
                    raise RuntimeError("No keyframes extracted from video.")

                frame_paths = [item.path for item in extracted_keyframes]
                frame_timestamps = [item.timestamp_sec for item in extracted_keyframes]
                frame_captions: list[str] = []
                face_count = 0

                clip_vectors = self.inference_service.get_clip_embeddings(frame_paths, batch_size=16)
                caption_tasks = [
                    asyncio.create_task(self.ollama_client.caption_image(frame_path))
                    for frame_path in frame_paths
                ]

                for frame_idx, (frame_path, clip_vector) in enumerate(zip(frame_paths, clip_vectors)):
                    frame_media_id = self._build_media_id(frame_path, prefix="frame")
                    frame_timestamp = frame_timestamps[frame_idx]
                    faces = self.inference_service.get_faces(frame_path)
                    caption = await caption_tasks[frame_idx]

                    frame_captions.append(caption)
                    face_count += len(faces)
                    self._upsert_clip_embedding(
                        media_id=frame_media_id,
                        vector=clip_vector,
                        path=str(video_path),
                        extra_metadata={
                            "frame_media_id": frame_media_id,
                            "frame_index": frame_idx,
                            "frame_timestamp_sec": frame_timestamp,
                        },
                    )
                    self._upsert_face_embeddings(
                        media_id=frame_media_id,
                        faces=faces,
                        path=str(video_path),
                        extra_metadata={
                            "frame_media_id": frame_media_id,
                            "frame_index": frame_idx,
                            "frame_timestamp_sec": frame_timestamp,
                        },
                    )
                    self.metadata_db.upsert_video_keyframe(
                        frame_media_id=frame_media_id,
                        video_media_id=video_media_id,
                        frame_index=frame_idx,
                        timestamp_sec=frame_timestamp,
                        frame_path=str(frame_path),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )

                    await asyncio.sleep(0)

                aggregated_caption = await self.ollama_client.summarize_video_captions(
                    frame_captions
                )
        except Exception:
            return False

        media = MediaFile(
            id=video_media_id,
            path=str(video_path.resolve()),
            hash=file_hash,
            caption=aggregated_caption,
            created_at=datetime.now(timezone.utc),
            metadata_json={
                "type": "video",
                "keyframes": len(frame_captions),
                "face_count_total": face_count,
            },
        )
        self.metadata_db.upsert_media(media)
        return True

    @staticmethod
    def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _build_media_id(path: Path, prefix: str = "media") -> str:
        return f"{prefix}_{path.stem}_{uuid.uuid4().hex[:12]}"

    def _upsert_clip_embedding(
        self,
        media_id: str,
        vector: list[float],
        path: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.vector_store.clip_collection is None:
            raise RuntimeError("Chroma clip collection is not initialized.")

        metadata = {"path": path}
        if extra_metadata:
            metadata.update(extra_metadata)
        self.vector_store.clip_collection.upsert(
            ids=[media_id],
            embeddings=[vector],
            metadatas=[metadata],
        )

    def _upsert_face_embeddings(
        self,
        media_id: str,
        faces: list[dict[str, Any]],
        path: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.vector_store.face_collection is None:
            raise RuntimeError("Chroma face collection is not initialized.")

        if not faces:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for face_idx, face in enumerate(faces):
            embedding = face.get("embedding")
            if not isinstance(embedding, list):
                continue
            ids.append(f"{media_id}_face_{face_idx}")
            embeddings.append([float(v) for v in embedding])
            metadatas.append(
                {
                    "path": path,
                    "parent_media_id": media_id,
                    "score": float(face.get("score", 0.0)),
                    "bbox": json.dumps(face.get("bbox", [])),
                }
            )
            if extra_metadata:
                metadatas[-1].update(extra_metadata)

        if ids:
            self.vector_store.face_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    @staticmethod
    def _extract_keyframes(
        video_path: Path,
        output_dir: Path,
        interval_sec: int = 3,
        scene_delta_threshold: float = 15.0,
    ) -> list[ExtractedKeyframe]:
        """Extract keyframes every N seconds and on significant scene changes."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 25.0
        interval_frames = max(1, int(fps * max(1, interval_sec)))

        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[ExtractedKeyframe] = []

        prev_gray: Any = None
        frame_index = 0
        saved_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_save = frame_index % interval_frames == 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                delta = cv2.absdiff(gray, prev_gray)
                mean_delta = float(delta.mean())
                if mean_delta >= scene_delta_threshold:
                    should_save = True

            if should_save:
                frame_file = output_dir / f"frame_{saved_index:06d}.jpg"
                written = cv2.imwrite(str(frame_file), frame)
                if written:
                    timestamp_sec = float(frame_index / fps)
                    frame_paths.append(
                        ExtractedKeyframe(path=frame_file, timestamp_sec=timestamp_sec)
                    )
                    saved_index += 1

            prev_gray = gray
            frame_index += 1

        capture.release()
        return frame_paths

