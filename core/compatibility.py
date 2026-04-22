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
    results.append(_check_ffmpeg_path())
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

    if ort_version.startswith("1.17."):
        return CheckResult(
            name="InsightFace / onnxruntime-gpu",
            status="PASS",
            details=f"onnxruntime-gpu={ort_version} (expected line: 1.17.x for CUDA 12.1).",
        )

    return CheckResult(
        name="InsightFace / onnxruntime-gpu",
        status="WARN",
        details=f"onnxruntime-gpu={ort_version}; recommended 1.17.x for CUDA 12.1.",
    )


def _check_ffmpeg_path() -> CheckResult:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return CheckResult(
            name="FFmpeg in PATH",
            status="FAIL",
            details="`ffmpeg` executable not found in PATH.",
        )

    try:
        proc = subprocess.run(
            [ffmpeg, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "ffmpeg found"
        return CheckResult(
            name="FFmpeg in PATH",
            status="PASS",
            details=first_line,
        )
    except Exception as exc:  # pragma: no cover
        return CheckResult(
            name="FFmpeg in PATH",
            status="WARN",
            details=f"`ffmpeg` found but version probe failed: {exc}",
        )


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False

