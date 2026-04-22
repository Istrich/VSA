# Stage 3 Change-Log (Implementer)

## What
- Added `core/indexer.py` with media orchestration pipeline:
  - recursive scanning by supported extensions,
  - SHA256 deduplication against SQLite `media.hash`,
  - image pipeline: CLIP -> FaceID -> Ollama caption,
  - video pipeline: keyframe extraction -> per-frame indexing -> caption aggregation.
- Added async `OllamaClient` (`httpx.AsyncClient`) for local Ollama API (`/api/generate`).
- Implemented required VLM prompt for image description (moondream2).

## Where
- `core/indexer.py`
- `docs/plans/2026-04-22-vsa-stage3-indexer-ollama.md`

## Why
- Build end-to-end indexing flow required for semantic archive population.

## How to Verify
1. Ensure dependencies and local Ollama are running.
2. Initialize indexer:
   - `indexer = MediaIndexer()`
3. Run:
   - `await indexer.index_directory("/path/to/media")`
4. Confirm:
   - new rows in SQLite `media`,
   - vectors in Chroma `embeddings_clip` / `embeddings_faces`,
   - video rows contain aggregated captions and keyframe metadata.
