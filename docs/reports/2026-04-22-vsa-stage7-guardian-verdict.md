# Guardian Verdict

VERDICT: APPROVE

## Reasons
- GPU path for InsightFace remains pinned to `CUDAExecutionProvider`.
- CLIP processing is upgraded from single-item flow to batched inference, reducing avoidable GPU underutilization.
- Hybrid score merge now normalizes CLIP and Face branches before combining, reducing branch domination risk.
- Video keyframes now persist timestamps in SQLite (`video_keyframes`), enabling time-localized retrieval outputs.
- `OLLAMA_NUM_PARALLEL` is now configurable and bounded by an async semaphore.
- Existing compatibility guardrails are preserved:
  - no Chroma collection rename/restructure,
  - no SillyTavern card format changes,
  - no `prompts.json` schema changes,
  - no Docker host-mode changes.

## Required Fixes
- None blocking within Stage 7 scope.

## Follow-up Recommendations
- Add optional PyAV/decord extractor behind a feature flag for faster decode on large video batches.
- Add integration test for `video_keyframes` persistence and timestamp projection into search results.
