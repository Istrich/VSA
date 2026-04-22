# Vision Semantic Archive — Stage 6 Plan

## Scope
Implement runtime compatibility checklist checks and expose them in UI:
- ChromaDB + SQLite co-location in `./data`.
- Ollama health-check validation.
- InsightFace dependency validation for `onnxruntime-gpu` and CUDA 12.1.
- FFmpeg availability in system PATH.

## Architecture Decisions
1. Add `core/compatibility.py` as a dedicated diagnostics module.
2. Keep checks lightweight and side-effect free:
   - no model downloads,
   - no media indexing.
3. Present results in Streamlit via a dedicated section and explicit re-run button.

## Risks
- Environment without optional dependencies may return partial diagnostics.
- Version strings may vary by package build.

## Migration
- No storage or schema migration required.

## Rollback
- Remove `core/compatibility.py` and UI checklist section.

## Implementation Steps
1. Implement checks for all checklist items.
2. Integrate diagnostics rendering in `streamlit_app.py`.
3. Validate syntax/lints and add workflow artifacts.
