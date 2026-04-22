"""Model downloader utilities for VSA Settings/Status tab."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from .models import ModelRegistry, ModelSpec

ProgressCallback = Callable[[float], None]


class ModelDownloader:
    """Downloads required model assets to local `./models` tree."""

    def __init__(self, root: str | Path = "./models") -> None:
        self.registry = ModelRegistry(root=root)

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
            raise RuntimeError(f"No download URL configured for model `{model_name}`.")

        target_path = self.registry.root / spec.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        download_target = target_path
        is_zip_archive = spec.download_url.lower().endswith(".zip")
        if is_zip_archive:
            download_target = target_path.parent.parent / f"{spec.name}.zip"
            download_target.parent.mkdir(parents=True, exist_ok=True)

        self._download_with_retries(
            url=spec.download_url,
            destination=download_target,
            progress_callback=progress_callback,
        )
        if spec.sha256:
            self._validate_sha256(download_target, spec.sha256)
        if is_zip_archive:
            extract_root = target_path.parent.parent
            self._extract_zip(download_target, extract_root)
            if download_target.exists():
                download_target.unlink()
        if not target_path.exists():
            raise RuntimeError(f"Downloaded model is missing expected file: {target_path}")
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
    ) -> None:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, 4):
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
                if temp_path.exists():
                    temp_path.unlink()
                if attempt == 3:
                    raise RuntimeError(f"Download failed after retries: {exc}") from exc

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
            raise RuntimeError(
                f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}."
            )

    @staticmethod
    def _extract_zip(zip_path: Path, destination_dir: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(destination_dir)

