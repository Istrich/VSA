# Vision Semantic Archive — Stage 3 Plan

## Scope
Implement media orchestration pipeline in `core/indexer.py`:
- Recursive scanner with hash deduplication.
- Image indexing pipeline (CLIP -> Face -> Ollama caption).
- Video indexing through keyframe extraction and per-frame indexing.
- Async Ollama API integration via `httpx`.

## Architecture Decisions
1. `OllamaClient` (async)
   - Uses `httpx.AsyncClient` against `http://localhost:11434`.
   - Supports image captioning (`moondream2`) and text summarization (`llama3`).
2. `MediaIndexer`
   - Coordinates DB, vector store, and inference service.
   - Hash-based skip using SQLite metadata table.
   - Stores frame-level results and aggregates a single video caption.
3. Video strategy
   - Extract keyframes every `N` seconds.
   - Optional scene-change trigger via grayscale frame delta threshold.
   - Index each keyframe as image-like item, then aggregate captions.

## Risks
- Ollama model availability/timeouts on local host.
- Large videos producing too many frames and slow throughput.
- Chroma add/upsert mismatches if ids are reused.

## Migration
- No metadata schema migration in this stage.
- Uses existing `media` table and Chroma collections.

## Rollback
- Remove `core/indexer.py` and Stage 3 documentation files.

## Implementation Steps
1. Add async Ollama client with robust error handling.
2. Implement scanner + hashing + media type dispatch.
3. Implement image pipeline and vector persistence.
4. Implement keyframe extraction and video caption aggregation.
5. Run syntax/lint checks and produce workflow artifacts.
