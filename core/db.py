"""Database adapters for metadata (SQLite) and vectors (ChromaDB)."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .models import MediaFile

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError:  # pragma: no cover - optional dependency in early bootstrap
    chromadb = None
    Collection = Any  # type: ignore[assignment]


class SQLiteMetadataDB:
    """SQLite metadata storage with FTS5 index for captions."""

    def __init__(self, db_path: str | Path = "./data/metadata.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def initialize(self) -> None:
        """Create base tables, indexes and FTS5 sync triggers."""
        with self._connect() as conn:
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

    def upsert_media(self, media: MediaFile) -> None:
        """Insert or update media metadata record."""
        try:
            metadata_json = json.dumps(media.metadata_json, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("media.metadata_json must be JSON-serializable.") from exc

        with self._connect() as conn:
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
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM media WHERE hash = ? LIMIT 1;", (file_hash,)
            ).fetchone()
            return row is not None

    def search_captions(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Keyword search over caption index through FTS5."""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        with self._connect() as conn:
            rows = conn.execute(
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
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE media SET path = ? WHERE hash = ? AND path != ?;",
                (new_path, file_hash, new_path),
            )
            return cursor.rowcount > 0

    def get_media_by_paths(self, paths: list[str]) -> list[dict[str, Any]]:
        """Return metadata rows for a list of absolute file paths."""
        if not paths:
            return []

        placeholders = ",".join("?" for _ in paths)
        with self._connect() as conn:
            rows = conn.execute(
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
        with self._connect() as conn:
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
        with self._connect() as conn:
            row = conn.execute(
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

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Build conservative FTS5 query from user text."""
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return ""
        escaped = [f"\"{token.replace('\"', '\"\"')}\"" for token in tokens]
        return " AND ".join(escaped)


class ChromaVectorStore:
    """Thin wrapper around ChromaDB collections used by VSA."""

    def __init__(self, persist_directory: str | Path = "./data/chroma") -> None:
        if chromadb is None:
            raise RuntimeError(
                "ChromaDB is not installed. Install package `chromadb` to enable vector storage."
            )
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.clip_collection: Collection | None = None
        self.face_collection: Collection | None = None

    def initialize(self) -> None:
        """Create/get all required collections."""
        self.clip_collection = self.client.get_or_create_collection(
            name="embeddings_clip", metadata={"dimension": 768}
        )
        self.face_collection = self.client.get_or_create_collection(
            name="embeddings_faces", metadata={"dimension": 512}
        )

