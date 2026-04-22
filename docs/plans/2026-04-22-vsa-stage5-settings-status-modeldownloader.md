# Vision Semantic Archive — Stage 5 Plan

## Scope
Implement model management UI and downloader flow:
- Add `Settings/Status` tab in Streamlit.
- Show required model statuses as `FOUND/MISSING`.
- Add `Download` button for missing models.
- Implement `ModelDownloader` using `urlretrieve` with progress reporting.

## Architecture Decisions
1. Extend model registry entries with download URL metadata.
2. Create dedicated `core/model_downloader.py`.
3. Reorganize UI into tabs:
   - `Search`
   - `Settings/Status`
4. Keep model paths aligned with requirements:
   - `./models/vision/...`
   - `./models/faces/...`

## Risks
- Remote URLs may be unavailable or require auth/rate limits.
- Large model files can make downloads long-running.

## Migration
- No schema/storage migration required.

## Rollback
- Remove model downloader module and tab-specific UI logic.

## Implementation Steps
1. Extend model specs in registry.
2. Implement downloader with progress callback.
3. Integrate status rendering and download actions into UI tab.
4. Run syntax/lint checks and produce workflow artifacts.
