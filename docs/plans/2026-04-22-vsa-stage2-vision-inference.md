# Vision Semantic Archive — Stage 2 Plan

## Scope
Implement `core/vision.py` for GPU-first visual inference:
- CLIP image embeddings via `open_clip` (`ViT-L-14`).
- Face detection/embeddings via `insightface` (`buffalo_l`).
- Runtime strictness for CUDA environment and VRAM hygiene.

## Architecture Decisions
1. `InferenceService` singleton
   - Single loaded copy of heavy models to reduce VRAM fragmentation.
2. CUDA policy
   - Hard fail if CUDA is unavailable.
   - Hard fail if `torch.version.cuda` is not `12.1.x`.
   - All model operations pinned to `device="cuda"`.
3. Face backend
   - `insightface.FaceAnalysis` with `providers=['CUDAExecutionProvider']`.
4. API
   - `get_clip_embedding(image_path) -> list[float]`
   - `get_faces(image_path) -> list[dict]` with bbox, score, embedding.
5. Warm-up
   - Run lightweight first pass for CLIP and FaceAnalysis to stabilize latency.

## Risks
- Environment mismatch (CUDA version/provider availability).
- Missing optional packages in dev environment.
- Large image sizes increasing VRAM usage.

## Migration
- No schema or storage migration required in Stage 2.

## Rollback
- Remove `core/vision.py` and related Stage 2 docs.

## Implementation Steps
1. Build service skeleton with singleton lifecycle.
2. Add dependency/runtime checks and model loading.
3. Implement CLIP and Face extraction methods.
4. Add warm-up and cache cleanup helpers.
5. Run syntax + lint checks.
