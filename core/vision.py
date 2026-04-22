"""GPU-first visual inference services for Vision Semantic Archive.

Changes vs. previous revision:

- Singleton flag is set only on the happy path, so a failed initialization no
  longer leaves the service in a permanent broken state (BUG-N05).
- ``max_vram_gb`` constructor argument was never consumed and was removed
  (BUG-N07).
- Per-call ``gc.collect() + torch.cuda.empty_cache()`` was dropped; cleanup
  now happens only in explicit batch-ends (PERF-N01).
- InsightFace root follows ``Settings.insightface_home`` (which already
  contains the ``models/<name>`` convention required by ``FaceAnalysis``),
  matching ``ModelDownloader`` layout (BUG-N04).
- CLIP weights default to ``hf-hub:openai/clip-vit-large-patch14`` which
  ``open_clip`` resolves against its own cache — we no longer try to load a
  HuggingFace ``model.safetensors`` into open_clip (BUG-N06).
"""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import Settings, get_settings
from .exceptions import InferenceError

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required for vision inference (`opencv-python`).") from exc

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open_clip_torch is required for CLIP embeddings.") from exc

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "insightface is required for face embeddings (`insightface`, `onnxruntime-gpu`)."
    ) from exc


LOGGER = logging.getLogger(__name__)


class InferenceService:
    """Long-lived service that owns CLIP / InsightFace models."""

    def __init__(
        self,
        settings: Settings | None = None,
        clip_model_name: str | None = None,
        clip_pretrained: str | None = None,
        face_model_name: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = LOGGER
        self._init_lock = threading.Lock()
        self._ready = False
        self._clip_model_name = clip_model_name or self._settings.clip_model_name
        self._clip_pretrained = clip_pretrained or self._settings.clip_pretrained
        self._face_model_name = face_model_name or self._settings.face_model_name
        self.device: str = "cpu"
        self.clip_model: Any = None
        self.clip_preprocess: Any = None
        self.face_analyzer: Any = None

    def _assert_cuda_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise InferenceError("CUDA is not available. VSA vision inference requires GPU.")
        cuda_version = torch.version.cuda
        if not cuda_version:
            raise InferenceError("Unable to detect CUDA runtime version from PyTorch.")
        pin = self._settings.cuda_runtime_pin
        if pin and not cuda_version.startswith(pin):
            raise InferenceError(
                f"Unsupported CUDA runtime: {cuda_version}. Required prefix: {pin}."
            )
        try:
            _ = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise InferenceError("Failed to initialize CUDA device 0.") from exc

    def ensure_ready(self) -> None:
        """Lazily initialize CLIP/InsightFace models on first real use."""
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self._initialize_models()
            self._ready = True  # set only on happy path

    def _initialize_models(self) -> None:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self._assert_cuda_runtime()
            self.device = "cuda"
        elif not self._settings.allow_cpu:
            raise InferenceError(
                "CUDA is unavailable and CPU fallback is disabled. "
                "Set VSA_ALLOW_CPU=1 (or settings.allow_cpu=True) for dev mode."
            )
        else:
            self.device = "cpu"
            self._logger.warning("VSA vision is running in CPU fallback mode.")

        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self._clip_model_name,
                pretrained=self._clip_pretrained,
                device=self.device,
            )
            self.clip_model.eval()

            insightface_home = self._settings.insightface_home
            insightface_home.mkdir(parents=True, exist_ok=True)
            providers = (
                ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            )
            self.face_analyzer = FaceAnalysis(
                name=self._face_model_name,
                providers=providers,
                root=str(insightface_home),
            )
            self.face_analyzer.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=(640, 640),
            )
            self._warm_up_models()
        except Exception:
            self.clip_model = None
            self.clip_preprocess = None
            self.face_analyzer = None
            raise

    def _warm_up_models(self) -> None:
        """Run lightweight warm-up passes to stabilize first-request latency."""
        if self.clip_model is None or self.face_analyzer is None:
            raise InferenceError("Inference models are not initialized.")
        image_size = self.clip_model.visual.image_size
        if isinstance(image_size, tuple):
            height, width = image_size
        else:
            height = width = int(image_size)

        with torch.inference_mode():
            dummy = torch.zeros((1, 3, height, width), device=self.device)
            _ = self.clip_model.encode_image(dummy)

        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.face_analyzer.get(dummy_frame)
        self._cleanup_vram()

    def _cleanup_vram(self) -> None:
        """Release cached VRAM. Only called at batch-ends, not per item."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_clip_embedding(self, image_path: str | Path) -> list[float]:
        """Return normalized CLIP embedding vector for a single image."""
        vectors = self.get_clip_embeddings([image_path], batch_size=1)
        return vectors[0]

    def get_clip_embeddings(
        self,
        image_paths: list[str | Path],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """Return normalized CLIP embedding vectors for a batch of images."""
        if not image_paths:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.ensure_ready()
        if self.clip_model is None or self.clip_preprocess is None:
            raise InferenceError("CLIP model is not initialized.")

        normalized_paths = [Path(p) for p in image_paths]
        for image_path in normalized_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        all_vectors: list[list[float]] = []
        try:
            with torch.inference_mode():
                for idx in range(0, len(normalized_paths), batch_size):
                    batch_paths = normalized_paths[idx : idx + batch_size]
                    tensors: list[torch.Tensor] = []
                    for image_path in batch_paths:
                        with Image.open(image_path) as image:
                            tensors.append(self.clip_preprocess(image.convert("RGB")))

                    batch_tensor = torch.stack(tensors, dim=0).to(self.device)
                    features = self.clip_model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    all_vectors.extend(features.detach().cpu().tolist())
            return all_vectors
        except Exception as exc:
            raise InferenceError("Failed to compute CLIP embeddings batch.") from exc
        finally:
            if len(normalized_paths) >= batch_size:
                self._cleanup_vram()

    def get_clip_text_embedding(self, text: str) -> list[float]:
        """Return normalized CLIP embedding vector for text query."""
        if not text.strip():
            raise ValueError("Text query cannot be empty for CLIP text embedding.")
        self.ensure_ready()
        if self.clip_model is None:
            raise InferenceError("CLIP model is not initialized.")

        try:
            tokens = open_clip.tokenize([text]).to(self.device)
            with torch.inference_mode():
                features = self.clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            raise InferenceError("Failed to compute CLIP text embedding.") from exc

    def get_faces(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Return face detections with bbox and 512-d embedding vectors."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.ensure_ready()
        if self.face_analyzer is None:
            raise InferenceError("Face analyzer is not initialized.")

        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise InferenceError(f"OpenCV failed to read image: {image_path}")

            faces = self.face_analyzer.get(image_bgr)
            results: list[dict[str, Any]] = []
            for face in faces:
                embedding = getattr(face, "normed_embedding", None)
                if embedding is None:
                    embedding = getattr(face, "embedding", None)
                if embedding is None:
                    continue
                bbox = face.bbox.tolist() if hasattr(face.bbox, "tolist") else list(face.bbox)
                vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                score = float(getattr(face, "det_score", 0.0))
                results.append(
                    {
                        "bbox": [float(v) for v in bbox],
                        "score": score,
                        "embedding": [float(v) for v in vector],
                    }
                )
            return results
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Failed to compute face embeddings for {image_path}.") from exc
