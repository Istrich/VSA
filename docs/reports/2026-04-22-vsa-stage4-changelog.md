# Stage 4 Change-Log (Implementer)

## What
- Added hybrid search engine in `core/search.py`:
  - text semantic branch via CLIP text embedding + Chroma `embeddings_clip`,
  - face branch via reference face embedding + Chroma `embeddings_faces`,
  - keyword branch via SQLite FTS5,
  - merged weighted score: `S = w1*ClipSim + w2*FaceSim + w3*FTS_Score`.
- Added Streamlit interface in `streamlit_app.py`:
  - text input,
  - optional face reference upload,
  - top-k + branch weights controls,
  - gallery cards with score breakdown and metadata.
- Extended support modules:
  - `core/vision.py`: added `get_clip_text_embedding`.
  - `core/db.py`: added `get_media_by_paths`.

## Where
- `core/search.py`
- `streamlit_app.py`
- `core/vision.py`
- `core/db.py`
- `docs/plans/2026-04-22-vsa-stage4-search-ui.md`

## Why
- Deliver user-facing hybrid retrieval flow requested in Stage 4.

## How to Verify
1. Ensure indexed data already exists in SQLite/Chroma.
2. Start UI: `streamlit run streamlit_app.py`.
3. Enter text query and optionally upload face reference.
4. Validate ranked gallery output with score decomposition and metadata.
