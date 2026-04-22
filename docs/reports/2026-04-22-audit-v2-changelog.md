# Audit v2 — Implementer change-log

Plan: `docs/plans/2026-04-22-audit-v2-fixes.md`

## Новые файлы

| Файл | Назначение |
|---|---|
| `core/config.py` | `pydantic-settings` Settings + `get_settings()` |
| `core/container.py` | Процесс-уровневый `ServiceContainer` (DI root) |
| `core/exceptions.py` | Типизированные исключения (`StorageError`, `InferenceError`, `OllamaError`, `IngestError`, `ModelAssetError`, `SearchError`, `ConfigError`) |
| `core/logging_config.py` | Централизованный `dictConfig` |
| `core/cli.py` | CLI `python -m core.cli {index\|doctor\|download-models\|ui}` |
| `tests/*` | pytest-сьют (FTS, dedup, rerank, downloader, indexer) |
| `.github/workflows/ci.yml` | ruff + black + mypy + pytest на Python 3.11/3.12 |
| `.env.example` | Документированный набор env-переменных |
| `README.md` | Quickstart, layout, конфиг, workflow |
| `Dockerfile`, `docker-compose.yml` | Скелет деплоя на CUDA 12.1 |
| `.python-version` | Закрепление Python 3.11 |

## Изменённые файлы

### `core/db.py` (BUG-N12, N13, N14, N15, N16, N10 support)
- `_sanitize_fts_query` переписан без бэкслешей в f-string expression → работает на Python 3.11.
- Thread-local cached connection, `PRAGMA synchronous=NORMAL`, `busy_timeout` применяются один раз на соединение.
- `ChromaVectorStore.clip_collection/face_collection` — `@property` с lazy init.
- Добавлены `get_path_by_hash`, `update_path_metadata`, `rebind_media_path` для sync ChromaDB метаданных при перемещении файлов.
- Константы коллекций читаются из `Settings`.

### `core/vision.py` (BUG-N05, N06, N07, PERF-N01, BUG-N04 via Settings)
- Singleton-флаг `_ready=True` ставится только на happy-path.
- `_cleanup_vram` больше не вызывается на каждом запросе; только в конце больших батчей CLIP.
- `max_vram_gb` удалён (не использовался).
- CLIP по умолчанию `hf-hub:openai/clip-vit-large-patch14` (open_clip сам скачает).
- InsightFace `root` берётся из `Settings.insightface_home`, совместимо с расположением, которое создаёт `ModelDownloader`.
- CUDA-pin теперь опционален (`Settings.cuda_runtime_pin`), по умолчанию выключен.
- Все ошибки — `InferenceError` (типизировано).

### `core/model_downloader.py` (BUG-N24, N26, SEC-N02)
- Скачивает полный `buffalo_l.zip` и распаковывает в `<insightface_home>/models/buffalo_l/`.
- Экспоненциальный backoff, `ModelAssetError` после исчерпания ретраев.
- Валидация sha256 и для архива, и для финального файла.
- Zip-распаковка отказывает в абсолютных путях и `..`.

### `core/models.py`
- `ModelSpec` получил `archive_sha256` и `required` поля.
- `ModelRegistry` больше не требует отдельный CLIP-safetensors (устранил orphaned asset).
- Добавлены pydantic-модели `IndexingStats`, переделан `SearchQuery` (теперь вложенный `SearchWeights`, а не dict).

### `core/indexer.py` (BUG-N08, N09, N10, PERF-05, TYPES-N02)
- Все sync-операции (`_hash_file`, CLIP batch, face detect, `cv2.imread`, `_extract_keyframes`) завёрнуты в `asyncio.to_thread`.
- Caption-задачи собираются через `asyncio.gather(..., return_exceptions=True)`; неудачный кадр больше не обваливает всё видео.
- `OllamaClient` → единый `httpx.AsyncClient` с `aclose()`, все тайминги из `Settings`.
- При `rebind_path_by_hash` одновременно обновляется Chroma-метаданные (path).
- Возвращает `IndexingStats(BaseModel)`, принимает `cancel_event` и `progress_callback`.
- Все ошибки индексации — `IngestError`; ошибки Ollama — `OllamaError`.

### `core/search.py` (BUG-N17, N18, N19, N21, TYPES-N01)
- Публичный API: `search(query: SearchQuery) -> list[SearchResult]` (поддерживает плоские kwargs как legacy-путь).
- Per-branch min-max нормализация **до** мёржа.
- `best_frame_timestamp_sec` выбирается по максимальному sim в ветви, а не по порядку прихода.
- `search_with_uploaded_face` использует `TemporaryDirectory` → нет tempfile leak.
- `SearchError` оборачивает неудачи пайплайна; логируется через `LOGGER.exception`.

### `core/compatibility.py` (BUG-N22, N23)
- Принимает `ServiceContainer` (или берёт синглтон), не создаёт второй Chroma-клиент.
- Добавлены проверки: ONNXRuntime providers (`CUDAExecutionProvider`), open_clip_torch версия, ffprobe.

### `streamlit_app.py` (BUG-N01, N02, N03, UX-N01)
- Все сервисы получаются из `ServiceContainer.get()` → один Chroma-клиент на процесс.
- Индексация идёт в `threading.Thread`, UI отображает progress и поддерживает cancel.
- Ready-guard на вкладке Index: предупреждает о недостающих моделях.
- Описание вкладки Settings говорит, что CLIP управляется open_clip (не вводит в заблуждение).
- Лимит загрузки face-ref файла через `Settings.max_upload_size_mb`.
- Использует `SearchResult` напрямую вместо `dict[str, Any]`.

### `pyproject.toml`, `requirements.txt`, `.gitignore`, `.python-version`
- `[build-system]` с `setuptools>=68`.
- Закреплены upper-bounds для всех рантайм-зависимостей.
- `python_requires = ">=3.11,<3.13"`.
- `.gitignore` расширен: `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.coverage`, `htmlcov`, `dist/`, `build/`.
- Удалён committed `__pycache__/`.

## Что не делал (future work)

- **BUG-N11** — агрегированная сущность «видео» в Chroma требует миграции схемы v1→v2 с rollback-скриптом. Отдельный план.
- **BUG-N20** — NLP parser через Ollama `llama3`. Фиче-флаг в отдельной итерации.
- **PERF-N02** — батчированный InsightFace. Оборачивание в `asyncio.to_thread` уже снимает блок event-loop'а; дальнейший x-скейлинг — отдельная оптимизация.
- **HZ-N11** — Prometheus-метрики `/metrics`. Вне периметра.

## Верификация (локальная, без GPU)

- `python3 -c "import ast; ast.parse(...)"` на всех `core/**.py`, `streamlit_app.py`, `tests/**.py` → OK.
- Ruff/Black/mypy/pytest планируются в GH Actions (локально недоступен pip из-за повреждённого Python 3.14 системного окружения).

## Как проверить

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q
ruff check .
black --check .
python -m core.cli doctor
streamlit run streamlit_app.py
```
