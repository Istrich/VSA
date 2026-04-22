"""Unit tests for the FTS5 query sanitizer (BUG-N12 and neighbours).

These tests also guarantee that ``core.db`` parses and imports cleanly on
Python 3.11 — the module previously used a PEP-701-only f-string construct
that would have raised ``SyntaxError`` on the officially supported Python.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.db import SQLiteMetadataDB


class TestSanitize:
    def test_plain_words_quoted_and_joined_with_and(self) -> None:
        out = SQLiteMetadataDB._sanitize_fts_query("big tree sunset")
        assert out == '"big" AND "tree" AND "sunset"'

    def test_unicode_tokens_survive(self) -> None:
        out = SQLiteMetadataDB._sanitize_fts_query("кот на дереве")
        assert out == '"кот" AND "на" AND "дереве"'

    def test_punctuation_is_stripped(self) -> None:
        out = SQLiteMetadataDB._sanitize_fts_query('a" OR b; --')
        assert out == '"a" AND "OR" AND "b"'

    def test_empty_input_returns_empty_string(self) -> None:
        assert SQLiteMetadataDB._sanitize_fts_query("") == ""
        assert SQLiteMetadataDB._sanitize_fts_query("   ??? --- *** ") == ""

    def test_quotes_are_doubled(self) -> None:
        # Regex keeps only \w+, so this is mostly a smoke test that no
        # backslash-in-expression path is triggered on 3.11.
        out = SQLiteMetadataDB._sanitize_fts_query('hello "world"')
        assert '"hello"' in out
        assert '"world"' in out


class TestEndToEndFts:
    def test_sanitized_query_is_valid_for_sqlite_fts5(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "fts.db")
        db.initialize()
        # FTS should not crash on typical user input:
        results = db.search_captions("any-word: with punctuation; really!", limit=5)
        assert isinstance(results, list)

    def test_poison_query_does_not_raise(self, tmp_path) -> None:
        db = SQLiteMetadataDB(db_path=tmp_path / "fts.db")
        db.initialize()
        # Previously these would bubble up OperationalError from FTS5.
        try:
            db.search_captions('- "OR" NOT AND :* ^foo', limit=3)
            db.search_captions('"""""', limit=3)
        except sqlite3.OperationalError as exc:
            pytest.fail(f"sanitizer allowed FTS5 syntax through: {exc}")
