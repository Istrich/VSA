# Stage 2 Change-Log (Implementer)

## What
- Added `core/vision.py` with `InferenceService` singleton for GPU inference.
- Integrated CLIP via `open_clip`:
  - model `ViT-L-14`
  - pretrained default `openai`
  - strict `device='cuda'`
- Integrated FaceID via `insightface`:
  - model `buffalo_l`
  - providers `['CUDAExecutionProvider']`
- Added API methods:
  - `get_clip_embedding(image_path)`
  - `get_faces(image_path)`
- Added CUDA runtime checks and model warm-up.
- Added VRAM cleanup after heavy inference (`torch.cuda.empty_cache()` + `gc.collect()`).

## Where
- `core/vision.py`
- `docs/plans/2026-04-22-vsa-stage2-vision-inference.md`

## Why
- Provide the required visual inference foundation for indexing and hybrid search.

## How to Verify
1. Install required packages (`torch`, `open_clip_torch`, `insightface`, `onnxruntime-gpu`, `opencv-python`, `Pillow`, `numpy`).
2. Ensure CUDA runtime in PyTorch is `12.1.x`.
3. Run a smoke snippet:
   - `svc = InferenceService()`
   - `svc.get_clip_embedding("path/to/image.jpg")`
   - `svc.get_faces("path/to/image.jpg")`
4. Confirm returned vectors and no initialization/runtime exceptions.
