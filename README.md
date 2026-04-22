# Vision Semantic Archive (VSA)

Hybrid visual semantic archive with local CLIP embeddings, InsightFace face
recognition, SQLite FTS5 captions, ChromaDB vector storage and a Streamlit UI.

See `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` for the full architecture and
`docs/plans/` for active engineering plans.

## Requirements

- Python 3.11 or 3.12
- `ffmpeg` + `ffprobe` on PATH (for video keyframe extraction)
- Optional: NVIDIA GPU with CUDA 12.1 runtime for production-speed inference.
  CPU fallback is supported for development via `VSA_ALLOW_CPU=1`.
- Optional: [Ollama](https://ollama.com/) running on `localhost:11434` with
  `moondream2` (captions) and `llama3` (summaries) pulled locally.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e .[dev]
cp .env.example .env                # tune paths / URLs as needed

# Prefetch InsightFace buffalo_l pack (CLIP is downloaded automatically by
# open_clip on first use):
python -m core.cli download-models  # optional helper

streamlit run streamlit_app.py
```

On first startup the UI opens three tabs:

1. **Search** — hybrid CLIP + Face + FTS retrieval against what is already
   indexed.
2. **Index** — pick a directory and run ingestion in a background thread
   with live progress and cancel.
3. **Settings/Status** — download missing models, run compatibility checks.

## Configuration

All runtime knobs come from environment variables (or `.env`) handled by
`core/config.py`. Prefix everything with `VSA_`. The most common ones:

| Variable | Default | Purpose |
|---|---|---|
| `VSA_DATA_DIR` | `./data` | SQLite + ChromaDB location |
| `VSA_MODELS_DIR` | `./models` | Model asset root |
| `VSA_INSIGHTFACE_HOME` | `./models/insightface_home` | InsightFace pack root |
| `VSA_ALLOW_CPU` | `true` | Allow running without CUDA |
| `VSA_CUDA_RUNTIME_PIN` | _unset_ | If set, CUDA version prefix to require |
| `VSA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `VSA_OLLAMA_NUM_PARALLEL` | `1` | Concurrent VLM requests |
| `VSA_CLIP_PRETRAINED` | `hf-hub:openai/clip-vit-large-patch14` | open_clip spec |
| `VSA_MAX_UPLOAD_SIZE_MB` | `10` | Face reference upload limit |

See `.env.example` for the full list.

## Development

```bash
pip install -e .[dev]
ruff check .
black --check .
mypy core
pytest -q
```

Heavy native deps (torch, onnxruntime, opencv, insightface, chromadb) are
stubbed in `tests/conftest.py` for the pure-Python test suite. GPU-level
correctness is validated manually.

## Project layout

```
core/
  config.py            pydantic-settings Settings + get_settings()
  container.py         ServiceContainer: the only place that builds stores
  db.py                SQLiteMetadataDB + ChromaVectorStore
  exceptions.py        Typed VSA errors
  indexer.py           MediaIndexer + OllamaClient
  logging_config.py    dictConfig install
  model_downloader.py  ModelDownloader for InsightFace pack
  models.py            Pydantic data + ModelRegistry service
  search.py            HybridSearchEngine
  vision.py            InferenceService (CLIP + InsightFace)
  compatibility.py     run_compatibility_checks()
streamlit_app.py       Streamlit UI (Search / Index / Settings)
tests/                 pytest suite
docs/plans/            engineering plans (AIBox workflow artifacts)
docs/reports/          per-plan change-logs / verdicts / test reports
```

## Agent workflow

This repo follows the [AIBox](./docs/system-prompts/) role protocol:
Architect → Implementer → Guardian → Tester → Documenter. Every substantive
change lands as a plan in `docs/plans/` and a change-log + Guardian verdict
in `docs/reports/`.
