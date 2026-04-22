# Stage 5 Change-Log (Implementer)

## What
- Added `core/model_downloader.py` with class `ModelDownloader`:
  - downloads model files with `urllib.request.urlretrieve`,
  - supports progress callback for UI progress-bar updates.
- Extended `core/models.py`:
  - `ModelSpec` now includes `download_url`,
  - added `ModelRegistry.get_specs()`.
- Updated `streamlit_app.py`:
  - UI split into tabs `Search` and `Settings/Status`,
  - `Settings/Status` shows model `FOUND/MISSING`,
  - `Download` buttons for missing models with progress bar.

## Where
- `core/model_downloader.py`
- `core/models.py`
- `streamlit_app.py`
- `docs/plans/2026-04-22-vsa-stage5-settings-status-modeldownloader.md`

## Why
- Deliver required model management workflow directly in UI.

## How to Verify
1. Start app: `streamlit run streamlit_app.py`.
2. Open `Settings/Status` tab.
3. Check statuses for required models.
4. For any `MISSING`, click `Download` and verify progress + file creation under `./models`.
