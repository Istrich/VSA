"""Tests for SQLiteMetadataDB dedup / rebind / FTS triggers."""

from __future__ import annotations

from datetime import datetime, timezone

from core.db import SQLiteMetadataDB
from core.models import MediaFile


def _media(path: str, file_hash: str, caption: str | None = "hello tree") -> MediaFile:
    return MediaFile(
        id=f"media_{file_hash[:8]}",
        path=path,
        hash=file_hash,
        caption=caption,
        created_at=datetime.now(timezone.utc),
        metadata_json={"type": "image"},
    )


class TestDedup:
    def test_insert_then_exists_by_hash(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "h" * 64))
        assert db.media_exists_by_hash("h" * 64)
        assert not db.media_exists_by_hash("z" * 64)

    def test_get_path_by_hash_returns_current(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "h" * 64))
        assert db.get_path_by_hash("h" * 64) == "/a.jpg"

    def test_rebind_updates_path_and_returns_true(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "h" * 64))
        changed = db.rebind_path_by_hash(file_hash="h" * 64, new_path="/b.jpg")
        assert changed is True
        assert db.get_path_by_hash("h" * 64) == "/b.jpg"

    def test_rebind_noop_returns_false(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "h" * 64))
        changed = db.rebind_path_by_hash(file_hash="h" * 64, new_path="/a.jpg")
        assert changed is False


class TestFtsTriggers:
    def test_caption_search_finds_inserted(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "a" * 64, caption="big tree sunset"))
        db.upsert_media(_media("/b.jpg", "b" * 64, caption="small house"))
        rows = db.search_captions("tree", limit=5)
        paths = {row["path"] for row in rows}
        assert "/a.jpg" in paths
        assert "/b.jpg" not in paths

    def test_caption_search_respects_update_trigger(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "m.db")
        db.initialize()
        db.upsert_media(_media("/a.jpg", "a" * 64, caption="big tree sunset"))
        # Update: caption now has no "tree".
        db.upsert_media(_media("/a.jpg", "a" * 64, caption="small pond"))
        rows = db.search_captions("tree", limit=5)
        assert rows == []
        rows = db.search_captions("pond", limit=5)
        assert [row["path"] for row in rows] == ["/a.jpg"]
