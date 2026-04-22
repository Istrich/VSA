# Vision Semantic Archive — Stage 4 Plan

## Scope
Implement hybrid search engine and Streamlit user interface:
- Hybrid ranking over CLIP, FaceID, and FTS branches.
- Query inputs: text + optional face reference image.
- UI with search controls and result gallery with metadata.

## Architecture Decisions
1. `core/search.py`
   - `HybridSearchEngine` orchestrates 3 branches:
     - text semantic search in Chroma `embeddings_clip`,
     - face similarity search in Chroma `embeddings_faces`,
     - keyword search in SQLite FTS5.
   - Merge by media path and score with formula:
     - `S = w1 * ClipSim + w2 * FaceSim + w3 * FTS_Score`.
2. `streamlit_app.py`
   - Sidebar for weights/top-k.
   - Main form: text input + face reference uploader.
   - Results rendered as responsive gallery cards with scores and metadata.
3. DB extension
   - Add helper in SQLite adapter to resolve metadata rows by file paths.

## Risks
- Missing CLIP text encoder API could block text branch.
- Face reference may contain no detectable faces.
- Score normalization mismatch across branches.

## Migration
- No schema migrations in Stage 4.

## Rollback
- Remove `core/search.py`, `streamlit_app.py`, and Stage 4 docs.

## Implementation Steps
1. Extend `InferenceService` with CLIP text embedding.
2. Add metadata lookup helper in `SQLiteMetadataDB`.
3. Implement `HybridSearchEngine` with branch normalization and merge.
4. Build Streamlit search UI and gallery rendering.
5. Run syntax/lint checks and produce workflow artifacts.
