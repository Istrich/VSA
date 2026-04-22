# Tester Report

## Checked
- Syntax validation of `core/search.py` and `streamlit_app.py`.
- Lint diagnostics for modified files.
- Static review of hybrid score merge and UI flow.

## Reproduction Steps
1. Ensure Python deps are installed (`streamlit`, `chromadb`, `torch`, `open_clip_torch`, `insightface`, etc.).
2. Ensure Stage 1-3 data exists in local DB/vector store.
3. Run:
   - `streamlit run streamlit_app.py`
4. In UI:
   - enter text query,
   - optionally upload face reference image,
   - click `Run Search`.
5. Validate:
   - results are ranked,
   - each card shows path/caption/scores/metadata.

## Result
- Compile and lint checks passed.
- Runtime behavior requires local GPU/model/data environment.
