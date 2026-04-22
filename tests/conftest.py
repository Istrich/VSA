"""Shared pytest fixtures.

``core.vision`` imports CUDA-only libraries at module import time (torch,
open_clip, insightface). To keep the pure-Python part of the test suite
runnable without those packages we install lightweight stubs *before* any
``core.*`` import. CI installs the real packages, so the stubs become
no-ops there.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest


class _NoopContext:
    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


def _install_stub(name: str, attrs: dict[str, object] | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _ensure_stub(name: str, attrs: dict[str, object] | None = None) -> None:
    if name in sys.modules:
        return
    _install_stub(name, attrs)


def _bootstrap_heavy_stubs() -> None:
    """Install stubs for heavy native deps when they are absent."""
    _ensure_stub(
        "torch",
        {
            "cuda": types.SimpleNamespace(
                is_available=lambda: False, empty_cache=lambda: None
            ),
            "version": types.SimpleNamespace(cuda=None),
            "inference_mode": lambda: _NoopContext(),
            "zeros": lambda *a, **kw: None,
            "stack": lambda tensors, dim=0: None,
            "__version__": "0.0.0-stub",
            "Tensor": type("Tensor", (), {}),
        },
    )
    _ensure_stub(
        "open_clip",
        {
            "create_model_and_transforms": lambda *a, **kw: (None, None, None),
            "tokenize": lambda texts: None,
        },
    )
    _ensure_stub(
        "cv2",
        {
            "imread": lambda *a, **kw: None,
            "VideoCapture": lambda *a, **kw: _NoopContext(),
            "cvtColor": lambda *a, **kw: None,
            "absdiff": lambda *a, **kw: None,
            "imwrite": lambda *a, **kw: False,
            "COLOR_BGR2GRAY": 0,
            "CAP_PROP_FPS": 5,
        },
    )
    if "insightface" not in sys.modules:
        insightface = _install_stub("insightface")
        app_module = types.ModuleType("insightface.app")
        app_module.FaceAnalysis = type("FaceAnalysis", (), {})  # type: ignore[attr-defined]
        insightface.app = app_module  # type: ignore[attr-defined]
        sys.modules["insightface.app"] = app_module
    if "numpy" not in sys.modules:
        _install_stub(
            "numpy",
            {
                "zeros": lambda *a, **kw: None,
                "uint8": int,
            },
        )
    if "chromadb" not in sys.modules:
        chroma_stub = _install_stub(
            "chromadb",
            {"PersistentClient": lambda path=None, **kw: None},
        )
        api_pkg = types.ModuleType("chromadb.api")
        models_pkg = types.ModuleType("chromadb.api.models")
        collection_module = types.ModuleType("chromadb.api.models.Collection")
        collection_module.Collection = type(  # type: ignore[attr-defined]
            "Collection", (), {}
        )
        models_pkg.Collection = collection_module  # type: ignore[attr-defined]
        sys.modules["chromadb.api"] = api_pkg
        sys.modules["chromadb.api.models"] = models_pkg
        sys.modules["chromadb.api.models.Collection"] = collection_module
        chroma_stub.api = api_pkg  # type: ignore[attr-defined]


_bootstrap_heavy_stubs()


@pytest.fixture(autouse=True)
def _vsa_tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point VSA at a scratch directory and reset cached settings/container."""
    monkeypatch.setenv("VSA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VSA_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VSA_INSIGHTFACE_HOME", str(tmp_path / "models" / "insightface_home"))
    monkeypatch.setenv("VSA_ALLOW_CPU", "1")

    from core.config import get_settings

    get_settings.cache_clear()
    try:
        from core.container import ServiceContainer

        ServiceContainer.reset()
    except ImportError:  # pragma: no cover
        pass
    yield
    get_settings.cache_clear()


@pytest.fixture
def disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block any outbound network for tests that should be fully local."""
    import httpx

    def _forbid(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network calls are forbidden in this test")

    monkeypatch.setattr(httpx, "get", _forbid)
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: _forbid())


@pytest.fixture(autouse=True)
def _clean_os_vars() -> Iterator[None]:
    """Remove cached VSA env flags that might leak between tests."""
    yield
    for var in list(os.environ):
        if var.startswith("VSA_") and var not in {"VSA_ALLOW_CPU"}:
            os.environ.pop(var, None)
