# Stage 6 Change-Log (Implementer)

## What
- Added `core/compatibility.py` with checklist validators:
  - ChromaDB + SQLite co-location and initialization in `./data`,
  - Ollama health-check via `http://localhost:11434/api/tags`,
  - `onnxruntime-gpu` presence/version validation (recommended 1.17.x for CUDA 12.1),
  - FFmpeg availability in PATH and version probe.
- Extended `streamlit_app.py` with `Compatibility Checklist` section and run button.

## Where
- `core/compatibility.py`
- `streamlit_app.py`
- `docs/plans/2026-04-22-vsa-stage6-compatibility-checklist.md`

## Why
- Enforce runtime readiness checklist before indexing/search operations.

## How to Verify
1. Run UI: `streamlit run streamlit_app.py`.
2. Open `Compatibility Checklist` and click `Run Compatibility Checks`.
3. Confirm statuses PASS/WARN/FAIL with diagnostic details.
