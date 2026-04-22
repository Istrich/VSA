"""Runtime compatibility checks for VSA deployment.

The checks take the process-wide ``ServiceContainer`` so they never create a
second ``chromadb.PersistentClient`` for the same directory (BUG-N22). They
are safe to call from the Streamlit Settings/Status tab.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .config import Settings, get_settings

if TYPE_CHECKING:
    from .container import ServiceContainer


@dataclass
class CheckResult:
    """One checklist item result."""

    name: str
    status: str
    details: str


def run_compatibility_checks(
    container: ServiceContainer | None = None,
    settings: Settings | None = None,
) -> list[CheckResult]:
    """Run all stage checklist checks.

    If ``container`` is None the checks fetch the process-wide container so
    Chroma is not duplicated (BUG-N01/BUG-N22).
    """
    from .container import ServiceContainer as _Container

    effective_settings = settings or get_settings()
    effective_container = container or _Container.get(settings=effective_settings)

    results: list[CheckResult] = []
    results.append(_check_storage_co_location(effective_container, effective_settings))
    results.append(_check_ollama_health(effective_settings.ollama_base_url))
    results.append(_check_onnxruntime_gpu_cuda_compat())
    results.append(_check_onnxruntime_providers())
    results.append(_check_torch_cuda_runtime())
    results.append(_check_open_clip_version())
    results.append(_check_ffmpeg_path("ffmpeg"))
    results.append(_check_ffmpeg_path("ffprobe"))
    results.append(_check_model_files(effective_container))
    results.append(_check_chromadb_version())
    return results


def _check_storage_co_location(container: ServiceContainer, settings: Settings) -> CheckResult:
    base = settings.data_dir.resolve()
    sqlite_path = container.storage.metadata_db.db_path.resolve()
    chroma_path = container.storage.vector_store.persist_directory.resolve()

    sqlite_under = _is_under(sqlite_path, base)
    chroma_under = _is_under(chroma_path, base)

    sqlite_ready = False
    chroma_ready = False
    sqlite_error = ""
    chroma_error = ""

    try:
        container.storage.metadata_db.initialize()
        sqlite_ready = True
    except Exception as exc:  # pragma: no cover
        sqlite_error = str(exc)

    try:
        container.storage.vector_store.initialize()
        _ = container.storage.vector_store.clip_collection
        _ = container.storage.vector_store.face_collection
        chroma_ready = True
    except Exception as exc:  # pragma: no cover
        chroma_error = str(exc)

    ok = sqlite_under and chroma_under and sqlite_ready and chroma_ready
    if ok:
        return CheckResult(
            name="ChromaDB + SQLite co-located under data_dir",
            status="PASS",
            details=f"SQLite: {sqlite_path} | Chroma: {chroma_path}",
        )

    details = (
        f"sqlite_under_data={sqlite_under}, chroma_under_data={chroma_under}, "
        f"sqlite_ready={sqlite_ready}, chroma_ready={chroma_ready}"
    )
    if sqlite_error:
        details += f" | sqlite_error={sqlite_error}"
    if chroma_error:
        details += f" | chroma_error={chroma_error}"
    return CheckResult(
        name="ChromaDB + SQLite co-located under data_dir",
        status="FAIL",
        details=details,
    )


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
            details=f"onnxruntime-gpu={ort_version} (supported range: 1.17–1.20).",
        )
    return CheckResult(
        name="InsightFace / onnxruntime-gpu",
        status="WARN",
        details=f"onnxruntime-gpu={ort_version}; verify CUDA provider compatibility.",
    )


def _check_onnxruntime_providers() -> CheckResult:
    try:
        import onnxruntime as ort
    except ImportError:
        return CheckResult(
            name="ONNXRuntime providers",
            status="FAIL",
            details="onnxruntime package not importable.",
        )
    providers = list(ort.get_available_providers())
    has_cuda = "CUDAExecutionProvider" in providers
    status = "PASS" if has_cuda else "WARN"
    details = f"providers={providers}" + ("" if has_cuda else " (CPU fallback only)")
    return CheckResult(name="ONNXRuntime providers", status=status, details=details)


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
    details = (
        f"torch={torch.__version__}, cuda_available={cuda_available}, "
        f"cuda_version={cuda_version}"
    )
    if not cuda_available:
        details += " (CPU fallback required)"
    return CheckResult(name="PyTorch runtime", status=status, details=details)


def _check_open_clip_version() -> CheckResult:
    try:
        version = metadata.version("open_clip_torch")
    except metadata.PackageNotFoundError:
        return CheckResult(
            name="open_clip_torch",
            status="FAIL",
            details="Package `open_clip_torch` is not installed.",
        )
    return CheckResult(
        name="open_clip_torch",
        status="PASS",
        details=f"open_clip_torch={version}",
    )


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


def _check_model_files(container: ServiceContainer) -> CheckResult:
    status_map = container.model_registry.get_model_status()
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
