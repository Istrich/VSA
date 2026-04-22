# Vision Semantic Archive — Stage 1 Plan

## Scope
Implement the Stage 1 foundation:
- Project structure for core modules.
- Metadata storage in SQLite with FTS5 index for captions.
- Vector storage integration via ChromaDB.
- Pydantic entity definitions (minimum: `MediaFile`).
- Local model presence checks in `./models`.

## Architecture Decisions
1. `core/models.py`
   - Keep all shared entities in one place using Pydantic.
   - Add lightweight model registry/check API for UI integration (`FOUND/MISSING`).
2. `core/db.py`
   - `SQLiteMetadataDB` handles schema creation and CRUD/upsert operations.
   - `ChromaVectorStore` wraps Chroma collections:
     - `embeddings_clip` (768-d)
     - `embeddings_faces` (512-d)
3. FTS5 strategy
   - `media_fts` virtual table with triggers to keep index in sync with `media.caption`.
4. Safe defaults
   - Local relative paths by default (`./data/metadata.db`, `./data/chroma`).
   - Explicit error messages for missing optional dependencies (ChromaDB).

## Risks
- ChromaDB may be unavailable in runtime environment.
- SQLite build without FTS5 support (rare on modern Python builds).
- Future schema evolution can break consumers if unmanaged.

## Migration
- Initial schema only (no migration framework yet).
- `initialize()` is idempotent and can be rerun safely.

## Rollback
- Remove created `core/*` files.
- Drop generated local data paths (`./data/metadata.db`, `./data/chroma`) if needed.

## Implementation Steps
1. Create `core` package.
2. Add Pydantic entities and model status checker.
3. Add SQLite manager with FTS5 and media table schema.
4. Add ChromaDB wrapper with required collections.
5. Verify imports and basic initialization workflow.
