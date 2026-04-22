"""GPU-first visual inference services for Vision Semantic Archive."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
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


class InferenceService:
    """Singleton service that owns GPU vision models and inference routines."""

    _instance: InferenceService | None = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> InferenceService:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        face_model_name: str = "buffalo_l",
        max_vram_gb: int = 18,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.max_vram_gb = max_vram_gb
        self.device = "cuda"

        self._assert_cuda_runtime()

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            device=self.device,
        )
        self.clip_model.eval()

        self.face_analyzer = FaceAnalysis(
            name=face_model_name,
            providers=["CUDAExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        self._warm_up_models()

    def _assert_cuda_runtime(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. VSA vision inference requires GPU.")

        cuda_version = torch.version.cuda
        if not cuda_version:
            raise RuntimeError("Unable to detect CUDA runtime version from PyTorch.")

        if not cuda_version.startswith("12.1"):
            raise RuntimeError(
                f"Unsupported CUDA runtime: {cuda_version}. Required: 12.1.x."
            )

        try:
            device_name = torch.cuda.get_device_name(0)
            _ = device_name  # Explicitly touch GPU to fail fast on bad runtime state.
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to initialize CUDA device 0.") from exc

    def _warm_up_models(self) -> None:
        """Run lightweight warm-up passes to stabilize first-request latency."""
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
        gc.collect()
        torch.cuda.empty_cache()

    def get_clip_embedding(self, image_path: str | Path) -> list[float]:
        """Return normalized CLIP embedding vector for an image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                features = self.clip_model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            raise RuntimeError(f"Failed to compute CLIP embedding for {image_path}.") from exc
        finally:
            self._cleanup_vram()

    def get_clip_text_embedding(self, text: str) -> list[float]:
        """Return normalized CLIP embedding vector for text query."""
        if not text.strip():
            raise ValueError("Text query cannot be empty for CLIP text embedding.")

        try:
            tokens = open_clip.tokenize([text]).to(self.device)
            with torch.inference_mode():
                features = self.clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            raise RuntimeError("Failed to compute CLIP text embedding.") from exc
        finally:
            self._cleanup_vram()

    def get_faces(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Return face detections with bbox and 512-d embedding vectors."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"OpenCV failed to read image: {image_path}")

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
        except Exception as exc:
            raise RuntimeError(f"Failed to compute face embeddings for {image_path}.") from exc
        finally:
            self._cleanup_vram()

