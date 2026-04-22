# Stage 1 Change-Log (Implementer)

## What
- Created project package `core/`.
- Added `core/models.py` with Pydantic entities (`MediaFile`, `Face`, `SearchQuery`) and `ProcessingStatus`.
- Added `ModelRegistry` to verify required models in `./models`.
- Added `core/db.py` with:
  - `SQLiteMetadataDB` (`media` table + FTS5 `media_fts` + sync triggers),
  - `ChromaVectorStore` with `embeddings_clip` and `embeddings_faces`.

## Where
- `core/__init__.py`
- `core/models.py`
- `core/db.py`
- `docs/plans/2026-04-22-vsa-stage1-architecture.md`

## Why
- Build mandatory storage/indexing foundation for upcoming indexer and hybrid search modules.

## How to Verify
1. Install dependencies (`pydantic`, `chromadb`).
2. Run a short bootstrap script:
   - create `SQLiteMetadataDB().initialize()`
   - create `ChromaVectorStore().initialize()`
   - print `ModelRegistry().get_model_status()`
3. Confirm creation of local paths:
   - `./data/metadata.db`
   - `./data/chroma/`
