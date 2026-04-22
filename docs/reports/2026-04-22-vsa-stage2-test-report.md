# Tester Report

## Checked
- Syntax validation for `core/vision.py`.
- Static review of:
  - CUDA runtime guard (`12.1.x`),
  - CUDA provider configuration for InsightFace,
  - public API methods for image vectors.

## Reproduction Steps
1. Activate Python 3.11+ environment with CUDA-enabled PyTorch.
2. Run `python3 -m compileall core`.
3. Run minimal smoke script:
   - `from core.vision import InferenceService`
   - `svc = InferenceService()`
   - `clip_vec = svc.get_clip_embedding("sample.jpg")`
   - `faces = svc.get_faces("sample.jpg")`
4. Validate:
   - CLIP vector is non-empty float list.
   - Face results contain `bbox`, `score`, `embedding`.

## Result
- Syntax checks passed locally.
- Runtime verification requires installed GPU dependencies and sample media.
