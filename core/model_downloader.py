"""Model downloader utilities for VSA Settings/Status tab.

Changes vs. previous revision:

- Downloads the full ``buffalo_l.zip`` pack and extracts it into
  ``<insightface_home>/models/buffalo_l/`` so ``FaceAnalysis`` actually finds
  it (BUG-N04/N24).
- ``_download_with_retries`` now uses exponential backoff (BUG-N26) and
  always cleans up ``.part`` files on failure.
- Optional sha256 validation via ``ModelSpec.sha256`` / ``archive_sha256``
  (SEC-N02). The previous default of ``sha256=None`` is preserved, so the
  check is opt-in until maintainers commit digests.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from .config import Settings, get_settings
from .exceptions import ModelAssetError
from .models import ModelRegistry, ModelSpec

ProgressCallback = Callable[[float], None]
LOGGER = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads required model assets to local model tree."""

    def __init__(
        self,
        root: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        resolved_root = Path(root) if root is not None else resolved_settings.insightface_home
        self.registry = ModelRegistry(root=resolved_root)

    def get_specs(self) -> tuple[ModelSpec, ...]:
        """Expose available model specifications."""
        return self.registry.get_specs()

    def get_status_map(self) -> dict[str, str]:
        """Return FOUND/MISSING map for all required models."""
        return self.registry.get_model_status()

    def download_model(
        self,
        model_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Download model artifact by registered model name."""
        spec = self._resolve_spec(model_name)
        if not spec.download_url.strip():
            raise ModelAssetError(f"No download URL configured for model `{model_name}`.")

        target_path = self.registry.root / spec.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        is_zip_archive = spec.download_url.lower().endswith(".zip")

        if is_zip_archive:
            download_target = self.registry.root / f"{spec.name}.zip"
            download_target.parent.mkdir(parents=True, exist_ok=True)
        else:
            download_target = target_path

        self._download_with_retries(
            url=spec.download_url,
            destination=download_target,
            progress_callback=progress_callback,
        )

        if is_zip_archive and spec.archive_sha256:
            self._validate_sha256(download_target, spec.archive_sha256)

        if is_zip_archive:
            extract_root = self.registry.root / "models"
            extract_root.mkdir(parents=True, exist_ok=True)
            self._extract_zip(download_target, extract_root)
            try:
                download_target.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass

        if spec.sha256 and target_path.exists():
            self._validate_sha256(target_path, spec.sha256)

        if not target_path.exists():
            raise ModelAssetError(
                f"Downloaded model is missing expected file: {target_path}"
            )

        if progress_callback is not None:
            progress_callback(1.0)
        return target_path

    def _resolve_spec(self, model_name: str) -> ModelSpec:
        for spec in self.registry.get_specs():
            if spec.name == model_name:
                return spec
        raise KeyError(f"Unknown model name: {model_name}")

    @staticmethod
    def _download_with_retries(
        url: str,
        destination: Path,
        progress_callback: ProgressCallback | None,
        max_attempts: int = 3,
    ) -> None:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", "0"))
                        downloaded = 0
                        with temp_path.open("wb") as output:
                            for chunk in response.iter_bytes():
                                if not chunk:
                                    continue
                                output.write(chunk)
                                downloaded += len(chunk)
                                if progress_callback is not None and total_size > 0:
                                    progress_callback(min(1.0, downloaded / total_size))
                shutil.move(str(temp_path), str(destination))
                return
            except httpx.HTTPError as exc:
                last_exc = exc
                LOGGER.warning(
                    "Download attempt %s/%s failed for %s: %s",
                    attempt,
                    max_attempts,
                    url,
                    exc,
                )
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:  # pragma: no cover
                        pass
                if attempt < max_attempts:
                    time.sleep(min(2 ** attempt, 8))
        raise ModelAssetError(f"Download failed after {max_attempts} attempts: {last_exc}")

    @staticmethod
    def _validate_sha256(path: Path, expected_sha256: str) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ModelAssetError(
                f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}."
            )

    @staticmethod
    def _extract_zip(zip_path: Path, destination_dir: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ModelAssetError(
                        f"Refusing to extract unsafe archive member: {member.filename}"
                    )
            archive.extractall(destination_dir)
