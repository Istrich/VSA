"""Database adapters for metadata (SQLite) and vectors (ChromaDB).

Two long-lived singletons are provided: ``SQLiteMetadataDB`` and
``ChromaVectorStore``. They are owned by the process-wide ``ServiceContainer``
(``core/container.py``) so we never end up with two ``chromadb.PersistentClient``
instances pointed at the same directory (BUG-N01).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .exceptions import StorageError
from .models import MediaFile

LOGGER = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError:  # pragma: no cover - optional dependency in early bootstrap
    chromadb = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment,misc]


class SQLiteMetadataDB:
    """SQLite metadata storage with FTS5 index for captions.

    Uses a thread-local cached connection; each thread opens on first use and
    the same connection is reused. WAL and busy_timeout are applied once per
    new connection, not on every query (BUG-N13).
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        self.db_path = Path(db_path) if db_path is not None else resolved_settings.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        self._local.conn = conn
        return conn

    def initialize(self) -> None:
        """Create base tables, indexes and FTS5 sync triggers (idempotent)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connect()
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS media (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL UNIQUE,
                        hash TEXT NOT NULL UNIQUE,
                        caption TEXT,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_media_hash ON media(hash);
                    CREATE INDEX IF NOT EXISTS idx_media_created_at ON media(created_at);

                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER NOT NULL
                    );
                    INSERT INTO schema_version(version)
                    SELECT 1
                    WHERE NOT EXISTS (SELECT 1 FROM schema_version);

                    CREATE TABLE IF NOT EXISTS video_keyframes (
                        frame_media_id TEXT PRIMARY KEY,
                        video_media_id TEXT NOT NULL,
                        frame_index INTEGER NOT NULL,
                        timestamp_sec REAL NOT NULL,
                        frame_path TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_video_keyframes_video_media_id
                        ON video_keyframes(video_media_id);

                    CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
                        caption,
                        content='media',
                        content_rowid='rowid'
                    );

                    CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media BEGIN
                        INSERT INTO media_fts(rowid, caption) VALUES (new.rowid, COALESCE(new.caption, ''));
                    END;

                    CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media BEGIN
                        INSERT INTO media_fts(media_fts, rowid, caption) VALUES('delete', old.rowid, COALESCE(old.caption, ''));
                    END;

                    CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON media BEGIN
                        INSERT INTO media_fts(media_fts, rowid, caption) VALUES('delete', old.rowid, COALESCE(old.caption, ''));
                        INSERT INTO media_fts(rowid, caption) VALUES (new.rowid, COALESCE(new.caption, ''));
                    END;
                    """
                )
            self._initialized = True

    def upsert_media(self, media: MediaFile) -> None:
        """Insert or update media metadata record."""
        self.initialize()
        try:
            metadata_json = json.dumps(media.metadata_json, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise StorageError("media.metadata_json must be JSON-serializable.") from exc

        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO media (id, path, hash, caption, created_at, metadata_json)
                VALUES (:id, :path, :hash, :caption, :created_at, :metadata_json)
                ON CONFLICT(hash) DO UPDATE SET
                    path=excluded.path,
                    caption=excluded.caption,
                    created_at=excluded.created_at,
                    metadata_json=excluded.metadata_json;
                """,
                {
                    "id": media.id,
                    "path": media.path,
                    "hash": media.file_hash,
                    "caption": media.caption,
                    "created_at": media.created_at.isoformat(),
                    "metadata_json": metadata_json,
                },
            )

    def media_exists_by_hash(self, file_hash: str) -> bool:
        """Return True when a media entry with this hash already exists."""
        self.initialize()
        row = self._connect().execute(
            "SELECT 1 FROM media WHERE hash = ? LIMIT 1;", (file_hash,)
        ).fetchone()
        return row is not None

    def get_path_by_hash(self, file_hash: str) -> str | None:
        """Return stored path for a given hash, or None if missing."""
        self.initialize()
        row = self._connect().execute(
            "SELECT path FROM media WHERE hash = ? LIMIT 1;", (file_hash,)
        ).fetchone()
        if row is None:
            return None
        return str(row["path"])

    def search_captions(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Keyword search over caption index through FTS5."""
        self.initialize()
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        rows = self._connect().execute(
            """
            SELECT
                m.id,
                m.path,
                m.caption,
                m.created_at,
                bm25(media_fts) AS fts_score
            FROM media_fts
            JOIN media m ON m.rowid = media_fts.rowid
            WHERE media_fts MATCH ?
            ORDER BY fts_score ASC
            LIMIT ?;
            """,
            (safe_query, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def rebind_path_by_hash(self, file_hash: str, new_path: str) -> bool:
        """Update stored path for an existing hash, return True if updated."""
        self.initialize()
        conn = self._connect()
        with conn:
            cursor = conn.execute(
                "UPDATE media SET path = ? WHERE hash = ? AND path != ?;",
                (new_path, file_hash, new_path),
            )
            return cursor.rowcount > 0

    def get_media_by_paths(self, paths: list[str]) -> list[dict[str, Any]]:
        """Return metadata rows for a list of absolute file paths."""
        if not paths:
            return []
        self.initialize()
        placeholders = ",".join("?" for _ in paths)
        rows = self._connect().execute(
            f"""
            SELECT id, path, caption, created_at, metadata_json
            FROM media
            WHERE path IN ({placeholders});
            """,
            paths,
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_video_keyframe(
        self,
        frame_media_id: str,
        video_media_id: str,
        frame_index: int,
        timestamp_sec: float,
        frame_path: str,
        created_at: str,
    ) -> None:
        """Insert or update one extracted keyframe linkage."""
        self.initialize()
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO video_keyframes (
                    frame_media_id,
                    video_media_id,
                    frame_index,
                    timestamp_sec,
                    frame_path,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(frame_media_id) DO UPDATE SET
                    video_media_id=excluded.video_media_id,
                    frame_index=excluded.frame_index,
                    timestamp_sec=excluded.timestamp_sec,
                    frame_path=excluded.frame_path,
                    created_at=excluded.created_at;
                """,
                (
                    frame_media_id,
                    video_media_id,
                    frame_index,
                    timestamp_sec,
                    frame_path,
                    created_at,
                ),
            )

    def get_best_keyframe_timestamp(self, video_path: str) -> float | None:
        """Return earliest keyframe timestamp for a video path."""
        self.initialize()
        row = self._connect().execute(
            """
            SELECT MIN(vk.timestamp_sec) AS timestamp_sec
            FROM video_keyframes vk
            JOIN media m ON m.id = vk.video_media_id
            WHERE m.path = ?;
            """,
            (video_path,),
        ).fetchone()
        if row is None or row["timestamp_sec"] is None:
            return None
        return float(row["timestamp_sec"])

    def close(self) -> None:
        """Close the thread-local connection if present (used by tests)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None  # type: ignore[attr-defined]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Build conservative FTS5 query from user text.

        Extracts unicode word tokens and wraps each into a quoted FTS5 phrase
        literal, escaping internal double quotes. Written without backslashes
        inside f-string expressions so it compiles on Python 3.11 (PEP 701
        landed in 3.12).
        """
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return ""
        dq = '"'
        escaped = [dq + token.replace(dq, dq + dq) + dq for token in tokens]
        return " AND ".join(escaped)


class ChromaVectorStore:
    """Thin wrapper around ChromaDB collections used by VSA."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        if chromadb is None:
            raise StorageError(
                "ChromaDB is not installed. Install package `chromadb` to enable vector storage."
            )
        resolved_settings = settings or get_settings()
        self.persist_directory = (
            Path(persist_directory)
            if persist_directory is not None
            else resolved_settings.chroma_path
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._settings = resolved_settings
        self._client: Any | None = None
        self._clip_collection: Collection | None = None
        self._face_collection: Collection | None = None
        self._init_lock = threading.Lock()

    def initialize(self) -> None:
        """Create/get all required collections (idempotent, thread-safe)."""
        if self._clip_collection is not None and self._face_collection is not None:
            return
        with self._init_lock:
            if self._clip_collection is not None and self._face_collection is not None:
                return
            if self._client is None:
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._clip_collection = self._client.get_or_create_collection(
                name=self._settings.chroma_clip_collection,
                metadata={"dimension": 768},
            )
            self._face_collection = self._client.get_or_create_collection(
                name=self._settings.chroma_faces_collection,
                metadata={"dimension": 512},
            )

    @property
    def clip_collection(self) -> Collection:
        """CLIP collection with lazy init."""
        if self._clip_collection is None:
            self.initialize()
        assert self._clip_collection is not None
        return self._clip_collection

    @property
    def face_collection(self) -> Collection:
        """Face collection with lazy init."""
        if self._face_collection is None:
            self.initialize()
        assert self._face_collection is not None
        return self._face_collection

    def update_path_metadata(
        self,
        collection_name: str,
        old_path: str,
        new_path: str,
    ) -> int:
        """Rewrite metadata.path for all records matching ``old_path``.

        Returns the number of rows updated. Keeps Chroma and SQLite in sync
        when a media file moves on disk (BUG-N10).
        """
        if collection_name == self._settings.chroma_clip_collection:
            collection = self.clip_collection
        elif collection_name == self._settings.chroma_faces_collection:
            collection = self.face_collection
        else:
            raise StorageError(f"Unknown Chroma collection: {collection_name}")

        fetched = collection.get(where={"path": old_path}, include=["metadatas"])
        ids = fetched.get("ids") or []
        metadatas = fetched.get("metadatas") or []
        if not ids:
            return 0
        updated_metadatas: list[dict[str, Any]] = []
        for meta in metadatas:
            if isinstance(meta, dict):
                new_meta = dict(meta)
            else:
                new_meta = {}
            new_meta["path"] = new_path
            updated_metadatas.append(new_meta)
        collection.update(ids=list(ids), metadatas=updated_metadatas)
        LOGGER.debug(
            "Rebound %d rows in %s: %s -> %s",
            len(ids),
            collection_name,
            old_path,
            new_path,
        )
        return len(ids)

    def rebind_media_path(self, old_path: str, new_path: str) -> int:
        """Update path metadata in both CLIP and Faces collections."""
        if old_path == new_path:
            return 0
        total = 0
        for name in (
            self._settings.chroma_clip_collection,
            self._settings.chroma_faces_collection,
        ):
            total += self.update_path_metadata(name, old_path, new_path)
        return total
