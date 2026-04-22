# Stage 7 Change-Log (Debugger)

## What
- Added CLIP batch inference API in `InferenceService`:
  - `get_clip_embeddings(image_paths, batch_size=16)` for GPU-efficient batch processing.
  - `get_clip_embedding(...)` now delegates to batch API.
- Updated video indexing pipeline in `MediaIndexer`:
  - keyframes now carry `timestamp_sec`,
  - CLIP embeddings are computed in batches instead of one frame at a time,
  - frame-level metadata (`frame_index`, `frame_timestamp_sec`, `frame_media_id`) is stored in vector metadatas,
  - frame-to-video timestamp linkage is persisted in SQLite table `video_keyframes`.
- Added Ollama parallelism control:
  - `OLLAMA_NUM_PARALLEL` env support in `OllamaClient`,
  - request-level async semaphore for bounded parallel calls.
- Improved hybrid ranking normalization in `HybridSearchEngine`:
  - Min-Max scaling for `clip_sim` and `face_sim` before weighted merge.
- Updated UI defaults in `streamlit_app.py`:
  - default weights to Face=0.6, CLIP=0.4, FTS=0.0,
  - show best frame timestamp when available.
- Added root `.cursorrules` with mandatory type hints/Pydantic rule.

## Where
- `core/vision.py`
- `core/indexer.py`
- `core/search.py`
- `core/db.py`
- `streamlit_app.py`
- `.cursorrules`

## Why
- Remove key throughput bottlenecks on RTX 3090.
- Prevent branch-score imbalance in hybrid retrieval.
- Preserve frame-time context for actionable video search results.
- Add operational control for VLM parallel requests.

## How to Verify
1. Run syntax check:
   - `python3 -m py_compile core/vision.py core/indexer.py core/search.py core/db.py streamlit_app.py`
2. Start app:
   - `streamlit run streamlit_app.py`
3. In Search UI, verify default sliders are `CLIP=0.4`, `Face=0.6`, `FTS=0.0`.
4. Index a video and verify keyframes persisted:
   - inspect SQLite `video_keyframes` table for `timestamp_sec`.
5. Execute hybrid query with text + face and confirm ranking remains stable after Min-Max normalization.
