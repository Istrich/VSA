"""Runtime compatibility checks for VSA deployment."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from .db import ChromaVectorStore, SQLiteMetadataDB
from .models import ModelRegistry


@dataclass
class CheckResult:
    """One checklist item result."""

    name: str
    status: str
    details: str


def run_compatibility_checks(data_dir: str | Path = "./data") -> list[CheckResult]:
    """Run all stage checklist checks."""
    results: list[CheckResult] = []
    results.append(_check_storage_co_location(data_dir))
    results.append(_check_ollama_health())
    results.append(_check_onnxruntime_gpu_cuda_compat())
    results.append(_check_torch_cuda_runtime())
    results.append(_check_ffmpeg_path("ffmpeg"))
    results.append(_check_ffmpeg_path("ffprobe"))
    results.append(_check_model_files())
    results.append(_check_chromadb_version())
    return results


def _check_storage_co_location(data_dir: str | Path) -> CheckResult:
    base = Path(data_dir).resolve()
    metadata_db = SQLiteMetadataDB()

    sqlite_path = metadata_db.db_path.resolve()
    chroma_path = Path("./data/chroma").resolve()

    sqlite_ok = _is_under(sqlite_path, base)
    chroma_ok = _is_under(chroma_path, base)

    sqlite_ready = False
    chroma_ready = False
    sqlite_error = ""
    chroma_error = ""

    try:
        metadata_db.initialize()
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("SELECT 1;")
        sqlite_ready = True
    except Exception as exc:  # pragma: no cover
        sqlite_error = str(exc)

    try:
        vector_store = ChromaVectorStore(persist_directory=chroma_path)
        vector_store.initialize()
        chroma_ready = True
    except Exception as exc:  # pragma: no cover
        chroma_error = str(exc)

    ok = sqlite_ok and chroma_ok and sqlite_ready and chroma_ready
    if ok:
        return CheckResult(
            name="ChromaDB + SQLite in ./data",
            status="PASS",
            details=f"SQLite: {sqlite_path} | Chroma: {chroma_path}",
        )

    details = (
        f"sqlite_under_data={sqlite_ok}, chroma_under_data={chroma_ok}, "
        f"sqlite_ready={sqlite_ready}, chroma_ready={chroma_ready}"
    )
    if sqlite_error:
        details += f" | sqlite_error={sqlite_error}"
    if chroma_error:
        details += f" | chroma_error={chroma_error}"
    return CheckResult(name="ChromaDB + SQLite in ./data", status="FAIL", details=details)


def _check_ollama_health(base_url: str = "http://localhost:11434") -> CheckResult:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        models = payload.get("models")
        model_count = len(models) if isinstance(models, list) else 0
        return CheckResult(
            name="Ollama health-check",
            status="PASS",
            details=f"{url} reachable, models={model_count}",
        )
    except Exception as exc:
        return CheckResult(
            name="Ollama health-check",
            status="WARN",
            details=f"{url} unreachable or invalid response: {exc}",
        )


def _check_onnxruntime_gpu_cuda_compat() -> CheckResult:
    try:
        ort_version = metadata.version("onnxruntime-gpu")
    except metadata.PackageNotFoundError:
        return CheckResult(
            name="InsightFace / onnxruntime-gpu",
            status="FAIL",
            details="Package `onnxruntime-gpu` is not installed.",
        )

    if ort_version.startswith(("1.17.", "1.18.", "1.19.", "1.20.")):
        return CheckResult(
            name="InsightFace / onnxruntime-gpu",
            status="PASS",
            details=f"onnxruntime-gpu={ort_version} (supported range: 1.17+).",
        )

    return CheckResult(
        name="InsightFace / onnxruntime-gpu",
        status="WARN",
        details=f"onnxruntime-gpu={ort_version}; verify CUDA provider compatibility for your host.",
    )


def _check_torch_cuda_runtime() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="PyTorch runtime",
            status="FAIL",
            details="Package `torch` is not installed.",
        )

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda or "unknown"
    status = "PASS" if cuda_available else "WARN"
    details = f"torch={torch.__version__}, cuda_available={cuda_available}, cuda_version={cuda_version}"
    if not cuda_available:
        details += " (CPU fallback required)"
    return CheckResult(name="PyTorch runtime", status=status, details=details)


def _check_ffmpeg_path(binary_name: str) -> CheckResult:
    binary = shutil.which(binary_name)
    if not binary:
        return CheckResult(
            name=f"{binary_name} in PATH",
            status="FAIL",
            details=f"`{binary_name}` executable not found in PATH.",
        )

    try:
        proc = subprocess.run(
            [binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "ffmpeg found"
        return CheckResult(
            name=f"{binary_name} in PATH",
            status="PASS",
            details=first_line,
        )
    except Exception as exc:  # pragma: no cover
        return CheckResult(
            name=f"{binary_name} in PATH",
            status="WARN",
            details=f"`{binary_name}` found but version probe failed: {exc}",
        )


def _check_model_files() -> CheckResult:
    registry = ModelRegistry()
    status_map = registry.get_model_status()
    missing = [name for name, status in status_map.items() if status != "FOUND"]
    if not missing:
        return CheckResult(
            name="Model files on disk",
            status="PASS",
            details="All required model files are present.",
        )
    return CheckResult(
        name="Model files on disk",
        status="WARN",
        details=f"Missing models: {', '.join(missing)}",
    )


def _check_chromadb_version() -> CheckResult:
    try:
        version = metadata.version("chromadb")
    except metadata.PackageNotFoundError:
        return CheckResult(
            name="ChromaDB package",
            status="FAIL",
            details="Package `chromadb` is not installed.",
        )
    return CheckResult(
        name="ChromaDB package",
        status="PASS",
        details=f"chromadb={version}",
    )


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False

