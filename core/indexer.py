"""Media indexing pipeline with async Ollama integration.

Changes vs. previous revision:

- All sync, IO- or CPU-heavy calls (``_hash_file``, CLIP batch, face detect,
  ``cv2.imread``, keyframe extraction) are wrapped in ``asyncio.to_thread``
  so they no longer block the event loop (BUG-N08).
- Per-frame caption tasks are now awaited via ``asyncio.gather`` with
  ``return_exceptions=True`` so a single failed frame does not discard the
  whole video and does not leave unawaited tasks (BUG-N09).
- Hash-based deduplication now synchronises BOTH SQLite and ChromaDB path
  metadata through ``ChromaVectorStore.rebind_media_path`` (BUG-N10).
- ``index_directory`` returns a typed ``IndexingStats`` instead of a dict
  (TYPES-N02).
- Ollama client takes dependencies from ``Settings``, can be cancelled via
  ``aclose``, and keeps a single ``httpx.AsyncClient`` (PERF-05).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

import cv2
import httpx

from .config import Settings, get_settings
from .db import ChromaVectorStore, SQLiteMetadataDB
from .exceptions import IngestError, OllamaError
from .models import IndexingStats, MediaFile
from .vision import InferenceService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MOONDREAM_PROMPT = (
    "Describe this image concisely, focusing on main subjects, their actions, "
    "and the environment. Mention specific brands or landmarks if visible."
)
LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[IndexingStats, Path], None]


@dataclass(frozen=True)
class ExtractedKeyframe:
    """Keyframe file plus its source timestamp in seconds."""

    path: Path
    timestamp_sec: float


class OllamaClient:
    """Async client for local Ollama API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        num_parallel: int | None = None,
        max_retries: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        self.base_url = (base_url or resolved_settings.ollama_base_url).rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds or resolved_settings.ollama_timeout_sec)
        self.num_parallel = max(1, int(num_parallel or resolved_settings.ollama_num_parallel))
        self.max_retries = max(1, int(max_retries or resolved_settings.ollama_max_retries))
        self._semaphore = asyncio.Semaphore(self.num_parallel)
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._settings = resolved_settings

    async def aclose(self) -> None:
        """Release the shared HTTP connection pool."""
        await self._client.aclose()

    async def _post_generate(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/api/generate"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                try:
                    response = await self._client.post(url, json=payload)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    last_exc = exc
                    LOGGER.warning(
                        "Ollama request failed (attempt %s/%s): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt == self.max_retries:
                        raise OllamaError(f"Ollama request failed after retries: {exc}") from exc
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise OllamaError("Ollama returned non-JSON response.") from exc

            text = data.get("response", "")
            if not isinstance(text, str):
                raise OllamaError("Ollama response payload does not contain text field.")
            return text.strip()

        raise OllamaError(f"Ollama request failed: {last_exc}")

    async def caption_image(
        self,
        image_path: str | Path,
        model: str | None = None,
        prompt: str = MOONDREAM_PROMPT,
    ) -> str:
        """Generate concise image caption via VLM."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for captioning: {path}")

        bytes_payload = await asyncio.to_thread(path.read_bytes)
        image_base64 = base64.b64encode(bytes_payload).decode("utf-8")
        payload = {
            "model": model or self._settings.ollama_caption_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }
        return await self._post_generate(payload)

    async def summarize_video_captions(
        self,
        captions: list[str],
        model: str | None = None,
    ) -> str:
        """Aggregate frame captions into one video-level summary."""
        non_empty = [c for c in captions if c and c.strip()]
        if not non_empty:
            return ""

        bullet_list = "\n".join(f"- {caption}" for caption in non_empty)
        prompt = (
            "You are summarizing keyframes from one video. "
            "Write one concise paragraph (max 80 words) that merges repeated details, "
            "mentions main subjects, actions and setting.\n\n"
            f"Frame descriptions:\n{bullet_list}"
        )
        payload = {
            "model": model or self._settings.ollama_summary_model,
            "prompt": prompt,
            "stream": False,
        }
        return await self._post_generate(payload)


class MediaIndexer:
    """Coordinates media scanning and indexing into metadata/vector stores."""

    def __init__(
        self,
        metadata_db: SQLiteMetadataDB | None = None,
        vector_store: ChromaVectorStore | None = None,
        inference_service: InferenceService | None = None,
        ollama_client: OllamaClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.metadata_db = metadata_db or SQLiteMetadataDB(settings=self._settings)
        self.vector_store = vector_store or ChromaVectorStore(settings=self._settings)
        self.inference_service = inference_service or InferenceService(settings=self._settings)
        self.ollama_client = ollama_client or OllamaClient(settings=self._settings)
        self.metadata_db.initialize()
        self.vector_store.initialize()

    async def index_directory(
        self,
        root_directory: str | Path,
        keyframe_interval_sec: int = 2,
        scene_delta_threshold: float = 15.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingStats:
        """Recursively index all supported media files under root."""
        root = Path(root_directory)
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")

        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ]
        stats = IndexingStats(total_candidates=len(candidates))

        for media_path in candidates:
            if cancel_event is not None and cancel_event.is_set():
                LOGGER.info("Indexing cancelled by user after %s files.", stats.indexed)
                break

            suffix = media_path.suffix.lower()
            try:
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
            except IngestError:
                LOGGER.exception("Ingest failed: %s", media_path)
                stats.failed += 1
                if progress_callback is not None:
                    progress_callback(stats, media_path)
                continue

            if ok is True:
                stats.indexed += 1
            elif ok is False:
                stats.failed += 1
            else:
                stats.skipped += 1

            if progress_callback is not None:
                progress_callback(stats, media_path)
        return stats

    async def _index_image_file(self, image_path: Path) -> bool | None:
        file_hash = await asyncio.to_thread(self._hash_file, image_path)
        if self.metadata_db.media_exists_by_hash(file_hash):
            await self._rebind_path_if_changed(file_hash=file_hash, new_path=str(image_path.resolve()))
            return None

        try:
            clip_vector = await asyncio.to_thread(
                self.inference_service.get_clip_embedding, image_path
            )
            faces = await asyncio.to_thread(self.inference_service.get_faces, image_path)
            caption = await self.ollama_client.caption_image(image_path)
        except Exception as exc:
            LOGGER.exception("Failed indexing image: %s", image_path)
            raise IngestError(f"image indexing failed: {image_path}") from exc

        media_id = self._build_media_id(file_hash=file_hash, prefix="image")
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
        file_hash = await asyncio.to_thread(self._hash_file, video_path)
        if self.metadata_db.media_exists_by_hash(file_hash):
            await self._rebind_path_if_changed(file_hash=file_hash, new_path=str(video_path.resolve()))
            return None

        video_media_id = self._build_media_id(file_hash=file_hash, prefix="video")
        try:
            with tempfile.TemporaryDirectory(prefix="vsa_keyframes_") as tmp_dir:
                extracted_keyframes = await asyncio.to_thread(
                    self._extract_keyframes,
                    video_path,
                    Path(tmp_dir),
                    keyframe_interval_sec,
                    scene_delta_threshold,
                )
                if not extracted_keyframes:
                    raise IngestError(f"No keyframes extracted: {video_path}")

                frame_paths = [item.path for item in extracted_keyframes]
                frame_timestamps = [item.timestamp_sec for item in extracted_keyframes]

                clip_vectors = await asyncio.to_thread(
                    self.inference_service.get_clip_embeddings,
                    frame_paths,
                    16,
                )

                caption_awaitables: list[Awaitable[str]] = [
                    self.ollama_client.caption_image(frame_path) for frame_path in frame_paths
                ]
                caption_results = await asyncio.gather(
                    *caption_awaitables, return_exceptions=True
                )

                frame_captions: list[str] = []
                face_count = 0
                successful_frames = 0
                for frame_idx, (frame_path, clip_vector) in enumerate(
                    zip(frame_paths, clip_vectors, strict=True)
                ):
                    caption_result = caption_results[frame_idx]
                    if isinstance(caption_result, BaseException):
                        LOGGER.warning(
                            "Caption failed for %s frame %s: %s",
                            video_path,
                            frame_idx,
                            caption_result,
                        )
                        caption_text = ""
                    else:
                        caption_text = caption_result
                        successful_frames += 1

                    frame_media_id = self._build_media_id(
                        file_hash=f"{file_hash}_{frame_idx}", prefix="frame"
                    )
                    frame_timestamp = frame_timestamps[frame_idx]
                    try:
                        faces = await asyncio.to_thread(
                            self.inference_service.get_faces, frame_path
                        )
                    except Exception:
                        LOGGER.exception(
                            "Face detection failed for %s frame %s", video_path, frame_idx
                        )
                        faces = []

                    frame_captions.append(caption_text)
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

                if successful_frames == 0:
                    raise IngestError(
                        f"All {len(frame_paths)} frame captions failed for {video_path}"
                    )

                try:
                    aggregated_caption = await self.ollama_client.summarize_video_captions(
                        frame_captions
                    )
                except OllamaError:
                    LOGGER.exception("Video summary failed; using first non-empty caption.")
                    aggregated_caption = next((c for c in frame_captions if c), "")
        except IngestError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed indexing video: %s", video_path)
            raise IngestError(f"video indexing failed: {video_path}") from exc

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

    async def _rebind_path_if_changed(self, file_hash: str, new_path: str) -> None:
        old_path = self.metadata_db.get_path_by_hash(file_hash)
        if old_path is None or old_path == new_path:
            return
        self.metadata_db.rebind_path_by_hash(file_hash=file_hash, new_path=new_path)
        try:
            await asyncio.to_thread(
                self.vector_store.rebind_media_path, old_path, new_path
            )
        except Exception:  # defensive; Chroma update is best-effort
            LOGGER.exception(
                "Failed to rebind Chroma metadata for hash=%s (%s -> %s)",
                file_hash,
                old_path,
                new_path,
            )

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
    def _build_media_id(file_hash: str, prefix: str = "media") -> str:
        return f"{prefix}_{file_hash[:20]}"

    def _upsert_clip_embedding(
        self,
        media_id: str,
        vector: list[float],
        path: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
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
            meta: dict[str, Any] = {
                "path": path,
                "parent_media_id": media_id,
                "score": float(face.get("score", 0.0)),
                "bbox": json.dumps(face.get("bbox", [])),
            }
            if extra_metadata:
                meta.update(extra_metadata)
            metadatas.append(meta)

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
        interval_sec: int = 2,
        scene_delta_threshold: float = 15.0,
    ) -> list[ExtractedKeyframe]:
        """Extract keyframes every N seconds and on significant scene changes."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IngestError(f"Cannot open video: {video_path}")

        try:
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
            return frame_paths
        finally:
            capture.release()
