"""Tests for ``ModelDownloader`` — SHA256, retries, zip safety (SEC-N02, BUG-N24)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.exceptions import ModelAssetError
from core.model_downloader import ModelDownloader
from core.models import ModelSpec


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


class _StubTransport(httpx.BaseTransport):
    def __init__(self, payload: bytes, fail_times: int = 0) -> None:
        self._payload = payload
        self._fail_times = fail_times
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self._fail_times:
            return httpx.Response(503)
        return httpx.Response(
            200,
            content=self._payload,
            headers={"content-length": str(len(self._payload))},
        )


def _patch_client(monkeypatch: pytest.MonkeyPatch, transport: httpx.BaseTransport) -> None:
    original_init = httpx.Client.__init__

    def patched_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)


class TestDownload:
    def test_non_zip_download_writes_file_and_verifies_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"weights-bytes"
        transport = _StubTransport(payload)
        _patch_client(monkeypatch, transport)

        downloader = ModelDownloader(root=tmp_path)
        downloader.registry.required_models = (
            ModelSpec(
                name="demo",
                relative_path="demo/weights.bin",
                download_url="https://example.invalid/weights.bin",
                sha256=_sha256(payload),
            ),
        )
        target = downloader.download_model("demo")
        assert target.exists()
        assert target.read_bytes() == payload

    def test_wrong_sha_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"weights-bytes"
        _patch_client(monkeypatch, _StubTransport(payload))
        downloader = ModelDownloader(root=tmp_path)
        downloader.registry.required_models = (
            ModelSpec(
                name="demo",
                relative_path="demo/weights.bin",
                download_url="https://example.invalid/weights.bin",
                sha256="deadbeef",
            ),
        )
        with pytest.raises(ModelAssetError):
            downloader.download_model("demo")

    def test_zip_extracts_to_models_subtree_matching_insightface_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_payload = _build_zip(
            {
                "buffalo_l/w600k_r50.onnx": b"onnx-weights",
                "buffalo_l/det_10g.onnx": b"det-weights",
            }
        )
        _patch_client(monkeypatch, _StubTransport(zip_payload))

        downloader = ModelDownloader(root=tmp_path)
        downloader.registry.required_models = (
            ModelSpec(
                name="insightface_buffalo_l",
                relative_path="models/buffalo_l/w600k_r50.onnx",
                download_url="https://example.invalid/buffalo_l.zip",
                archive_sha256=_sha256(zip_payload),
            ),
        )
        target = downloader.download_model("insightface_buffalo_l")
        # Must match FaceAnalysis(root=tmp_path, name="buffalo_l") layout.
        assert target == tmp_path / "models" / "buffalo_l" / "w600k_r50.onnx"
        assert target.exists()
        assert (tmp_path / "models" / "buffalo_l" / "det_10g.onnx").exists()
        # Archive is cleaned up.
        assert not (tmp_path / "insightface_buffalo_l.zip").exists()

    def test_retries_exhausted_raises_model_asset_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = _StubTransport(b"", fail_times=10)
        _patch_client(monkeypatch, transport)

        # Bypass the sleep so the test does not run for 14 seconds.
        import core.model_downloader as md_module

        monkeypatch.setattr(md_module.time, "sleep", lambda _seconds: None)

        downloader = ModelDownloader(root=tmp_path)
        downloader.registry.required_models = (
            ModelSpec(
                name="demo",
                relative_path="demo/x.bin",
                download_url="https://example.invalid/x",
            ),
        )
        with pytest.raises(ModelAssetError):
            downloader.download_model("demo")
        assert transport.calls == 3

    def test_zip_with_absolute_path_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unsafe_zip = _build_zip({"/etc/passwd": b"root:x:0:0"})
        _patch_client(monkeypatch, _StubTransport(unsafe_zip))
        downloader = ModelDownloader(root=tmp_path)
        downloader.registry.required_models = (
            ModelSpec(
                name="unsafe",
                relative_path="models/unsafe/x.onnx",
                download_url="https://example.invalid/unsafe.zip",
            ),
        )
        with pytest.raises(ModelAssetError):
            downloader.download_model("unsafe")
