"""Model downloader utilities for VSA Settings/Status tab."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

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

        def _hook(block_num: int, block_size: int, total_size: int) -> None:
            if progress_callback is None:
                return
            if total_size <= 0:
                progress_callback(0.0)
                return
            downloaded = block_num * block_size
            progress = min(1.0, downloaded / total_size)
            progress_callback(progress)

        urlretrieve(spec.download_url, target_path, _hook)
        if progress_callback is not None:
            progress_callback(1.0)
        return target_path

    def _resolve_spec(self, model_name: str) -> ModelSpec:
        for spec in self.registry.get_specs():
            if spec.name == model_name:
                return spec
        raise KeyError(f"Unknown model name: {model_name}")

