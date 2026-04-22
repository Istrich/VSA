"""Tests for ``core.config.Settings`` env/defaults behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Settings, get_settings


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VSA_OLLAMA_NUM_PARALLEL", "4")
    monkeypatch.setenv("VSA_OLLAMA_BASE_URL", "http://custom:9999")
    monkeypatch.setenv("VSA_DATA_DIR", str(tmp_path / "alt-data"))
    get_settings.cache_clear()
    s = Settings()
    assert s.ollama_num_parallel == 4
    assert s.ollama_base_url == "http://custom:9999"
    assert s.data_dir == (tmp_path / "alt-data")
    assert s.sqlite_path == (tmp_path / "alt-data" / s.sqlite_filename)


def test_chroma_path_derives_from_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSA_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("VSA_CHROMA_SUBDIR", "vectors")
    get_settings.cache_clear()
    s = Settings()
    assert s.chroma_path == (tmp_path / "d" / "vectors")


def test_num_parallel_clamped_to_valid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSA_OLLAMA_NUM_PARALLEL", "0")
    with pytest.raises(Exception):
        Settings()
