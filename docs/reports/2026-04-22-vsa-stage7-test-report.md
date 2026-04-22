# Tester Report

## Checked
- Python compile check for all modified files:
  - `core/vision.py`
  - `core/indexer.py`
  - `core/search.py`
  - `core/db.py`
  - `streamlit_app.py`
- Lint diagnostics for modified files (no new lint errors).
- Manual code-path verification for:
  - CLIP batch API usage in video indexing,
  - Min-Max normalization before hybrid weighted merge,
  - keyframe timestamp persistence in SQLite,
  - `OLLAMA_NUM_PARALLEL` ingestion and bounded parallel requests.

## Reproduction Steps
1. Export optional parallelism (example):
   - `export OLLAMA_NUM_PARALLEL=2`
2. Run app:
   - `streamlit run streamlit_app.py`
3. Index a folder with at least one video.
4. Verify SQLite keyframe table:
   - `SELECT frame_media_id, video_media_id, frame_index, timestamp_sec FROM video_keyframes LIMIT 20;`
5. Run a text+face query and inspect result cards:
   - confirm score columns render,
   - confirm `best_frame_timestamp_sec` appears when frame-hit metadata is available.

## Result
- Compile/lint checks passed.
- Runtime throughput improvement and ranking quality should be validated on representative media corpus (environment-dependent).
