# VSA — Audit v2: план правок (P0/P1/P2/P3)

Date: 2026-04-22
Role: Architect → далее Implementer → Guardian → Tester → Documenter
Scope: все пункты `N*` из свежего аудита (см. ответ Analyzer от 2026-04-22, v2).

Исходный аудит и базовая карта находок — `docs/plans/2026-04-22-project-audit.md` (v1). Этот документ описывает ровно те правки, которые закрывают пункты `BUG-N*/HZ-N*/PERF-N*/SEC-N*/UX-N*/TYPES-N*`, не упомянутые в v1 или оставшиеся нерешёнными.

---

## 1. Классификация задачи

Комбинированная: структурное изменение архитектуры (service container / DI) + багфиксы + рефакторинг без изменения поведения + документация. Проходим полный workflow: Architect → Implementer → Guardian → Tester → Documenter.

## 2. Границы / non-goals

В периметре:
- все P0/P1 из Analyzer v2;
- большинство P2 (unit-тесты, CI, логирование, `pydantic-settings`, pydantic-контракты, tempfile, VRAM);
- минимальный README + `.env.example` + Dockerfile-скелет.

Вне периметра (оставляем задачами future work, явно фиксируем):
- **BUG-N11** — агрегированная сущность «видео» в Chroma + миграция v1→v2 (нужна отдельная итерация с согласованием rollback-скрипта Chroma).
- **BUG-N20** — реализация NLP-parser через Ollama `llama3`.
- **HZ-N11** — Prometheus-метрики / `/metrics` endpoint.
- **PERF-N02** — батчированная детекция InsightFace (пока оборачиваем `asyncio.to_thread`, это снимает блок event-loop'а, но не даёт x-скейлинга).

## 3. Архитектурные решения

### 3.1. Service container (`core/container.py`)
Вместо двух независимых `@st.cache_resource` для `MediaIndexer` и `HybridSearchEngine` (BUG-N01) вводится `ServiceContainer`:

- владеет одним `ChromaVectorStore` и одним `SQLiteMetadataDB` (**shared storage**);
- владеет одним `InferenceService` (CLIP+Faces) и одним `OllamaClient`;
- предоставляет `indexer()` и `search_engine()` как factory-методы, передавая готовые зависимости в конструкторы.

UI и compatibility-чеки получают сервисы из контейнера. Это закрывает BUG-N01, BUG-N22, упрощает тестирование, соответствует guardrail «DI через `context.bot_data`» (в нашем случае — `st.cache_resource` для контейнера).

### 3.2. Конфиг (`core/config.py` на `pydantic-settings`)
Единая `Settings` с полями: `data_dir`, `models_dir`, `insightface_home`, `chroma_path`, `sqlite_path`, `ollama_base_url`, `ollama_num_parallel`, `allow_cpu`, `clip_model_name`, `clip_pretrained`, `face_model_name`, `log_level`. Загружается из env + `.env`. Все хард-коды путей/URL в коде переводятся на эти поля (HZ-N08).

### 3.3. Логирование (`core/logging_config.py`)
`configure_logging(level)` с `dictConfig`: форматтер, root-logger, отдельные уровни для httpx/chromadb. Вызывается в `streamlit_app.main()` и `python -m vsa` (HZ-N10).

### 3.4. Типизированные контракты (Pydantic)
- `HybridSearchEngine.search(query: SearchQuery) -> list[SearchResult]`.
- `MediaIndexer.index_directory(...) -> IndexingStats` (новый класс в `models.py`).
- UI преобразует `SearchResult`/`IndexingStats` в визуал (TYPES-N01/N02).

### 3.5. Исправление путей моделей
- CLIP: default `clip_pretrained = "hf-hub:openai/clip-vit-large-patch14"` (open_clip≥2.24 понимает), `ModelRegistry` больше не требует safetensors-файла (BUG-N06/N25). В UI карточка CLIP становится статусом «Pretrained via open_clip cache».
- InsightFace: settings.`insightface_home = ./models/insightface_home`. `FaceAnalysis(root=settings.insightface_home)` ищет в `<home>/models/<name>/…`. `ModelDownloader` распаковывает `buffalo_l.zip` в `<home>/models/buffalo_l/…` (BUG-N04/N24).
- Обе записи в `ModelRegistry` получают `sha256` (SEC-N02); SHA для `buffalo_l.zip` и для `w600k_r50.onnx` после распаковки.

### 3.6. Ingestion pipeline
- Все sync-вызовы (`_hash_file`, `get_clip_embedding(s)`, `get_faces`, `_extract_keyframes`, `cv2.imread`) в async-методах оборачиваются в `asyncio.to_thread(...)` (BUG-N08).
- Параллельные caption-таски собираются через `asyncio.gather(*tasks, return_exceptions=True)` с per-frame логом ошибок и частичным успехом (BUG-N09).
- `rebind_path_by_hash` → новый контракт `Storage.rebind_media_path(hash, old_path, new_path)`, который одновременно обновляет SQLite и Chroma (запрос `collection.get(where={"path": old_path})` → `collection.update(ids=..., metadatas=[{"path": new_path}]*n)`) (BUG-N10).
- `MediaIndexer` возвращает `IndexingStats` вместо `dict[str, int]`.

### 3.7. Search
- Нормализация per-branch **до** мерджа: в каждой ветви (CLIP/Face/FTS) рассчитываем `score_in_branch ∈ [0..1]` через min-max по этой же ветви и укладываем уже нормализованные значения в кандидатов (BUG-N17/N18).
- Если кандидат пришёл только из одной ветви, значения других ветвей остаются 0 (а не ложно-нормализуются).
- `search_with_uploaded_face` использует `tempfile.TemporaryDirectory(...)` — нет ручного `unlink`, нет leak (BUG-N21).
- `best_frame_timestamp_sec` выбирается «по максимальному `clip_sim`/`face_sim` в ветви», а не «по последнему совпадению» (BUG-N19).

### 3.8. SQLite / FTS
- `_sanitize_fts_query` переписан без бэкслешей в f-string-выражении (BUG-N12). Добавлен unit-тест.
- `SQLiteMetadataDB` получает thread-local cached connection (`threading.local()`), `PRAGMA synchronous=NORMAL`, инициализация применяется только при первом `_connect()` в потоке (BUG-N13/N15).
- `ChromaVectorStore.clip_collection/face_collection` через `@property` с lazy init (BUG-N16).

### 3.9. Vision
- `_initialized=True` ставится внутри `_initialize_models` только на happy-path (BUG-N05).
- `_cleanup_vram` больше **не** вызывается в per-call `finally`. Оставлен только в конце больших батчей (`get_clip_embeddings`), не в `get_clip_text_embedding`/`get_faces` (PERF-N01).
- `max_vram_gb` — удалено из API (не использовалось) (BUG-N07).

### 3.10. UI (`streamlit_app.py`)
- Получает сервисы из `get_container()`.
- Вкладка Index запускает индексацию в `threading.Thread`, общая статистика через `st.session_state`, цикл опроса через `st.empty()` + `time.sleep(0.5)` до завершения потока. Поддерживается отмена через `threading.Event` (BUG-N02, HZ-N12).
- Ready-guard: если `ModelRegistry.is_ready()` — False, Run Indexing показывает warning и не запускается (BUG-N03).
- Settings/Status: корректно скрывает `Download` для CLIP (управляется open_clip) (UX-N01).

### 3.11. Compatibility
- `run_compatibility_checks` принимает `ServiceContainer`, не создаёт собственные клиенты (BUG-N22).
- Добавлены: `CUDAExecutionProvider` availability для ORT (BUG-N23), smoke `FaceAnalysis(root=…).prepare(ctx_id=-1, det_size=(320,320))`, версия `open_clip_torch`.

## 4. Миграции и rollback

- **Schema SQLite v2**: +`PRAGMA synchronous=NORMAL` (in-place, rollback — дефолтный `FULL`, без потери данных).
- **Chroma**: коллекции не переименовываются, формат не меняется → rollback не нужен (guardrail соблюдён).
- **`./models/insightface_home`**: если у пользователя уже есть `./models/faces/buffalo_l/`, добавляем compatibility-чек, который это детектит и в UI предлагает перенести. Rollback — переместить обратно.
- **`ModelRegistry` contract**: убираем запись CLIP safetensors, добавляем `buffalo_l` pack. Для пользователей, уже скачавших старый файл, показываем мягкий warning (не ломаем процесс).

## 5. План доставки по этапам

| Этап | Содержимое | Артефакт Implementer |
|---|---|---|
| P0-A | BUG-N12 (f-string), packaging (HZ-N01..N04), `.gitignore`, `pyproject [build-system]` | change-log pt.1 |
| P0-B | ModelRegistry/ModelDownloader (BUG-N04/N24/N25, SEC-N02); vision.py (BUG-N05/N06, PERF-N01, BUG-N07) | change-log pt.2 |
| P1-A | `core/config.py`, `core/container.py`, `core/exceptions.py`, `core/logging_config.py`; рефакторинг `db.py` (BUG-N13..N16) | change-log pt.3 |
| P1-B | `indexer.py` (BUG-N08/N09/N10, IndexingStats); `search.py` (BUG-N17..N21, SearchResult); `compatibility.py` (BUG-N22/N23) | change-log pt.4 |
| P1-C | `streamlit_app.py` → container + threading + progress + ready-guard | change-log pt.5 |
| P2-A | tests/ (pytest), CI (GH Actions), ruff/black/mypy | change-log pt.6 |
| P2-B | README.md, `.env.example`, синхронизация `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` | change-log pt.7 |
| P3 | Dockerfile-скелет, `python -m vsa` CLI | change-log pt.8 |

## 6. Критерии готовности (DoD)

1. Все модули компилируются на Python 3.11 (AST-check через `python -m py_compile`).
2. `pytest -q` зелёный локально; CI на GH Actions зелёный.
3. `ruff check .` и `black --check .` чистые.
4. `mypy core/` без новых ошибок (относительно baseline; при необходимости фиксируем baseline).
5. `streamlit run streamlit_app.py` стартует без impacts на Settings/Index/Search (даже без GPU — CPU-fallback).
6. В Settings-вкладке `Run Compatibility Checks` проходит без создания второго Chroma-клиента.
7. Во вкладке Index отображается прогресс и поддерживается отмена.
8. Документация обновлена: README, `.env.example`, `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` синхронизирован.
9. Чек-лист `docs/reports/2026-04-22-audit-v2-*.md` заполнен (changelog + Guardian verdict + test report + documentation update).

## 7. Риски и mitigations

| Риск | Mitigation |
|---|---|
| Переход CLIP на `hf-hub:...` требует open_clip≥2.24 и сети при первом запуске | Пина версии в `pyproject.toml`; в offline-окружении остаётся путь через `clip_pretrained` из settings (env override) |
| Shared storage в `@st.cache_resource` ломает hot-reload при правках кода | Документируем перезапуск Streamlit при смене схемы; тест ручного сценария |
| `threading.Thread` в Streamlit требует аккуратной работы с `st.session_state` | Используем чистый dict-holder, UI читает только из него, поток не вызывает st-API напрямую |
| Удаление `_cleanup_vram` per-call может поднять VRAM-peak | Оставляем cleanup в конце больших батчей и после OOM-exception handler; риск принимаем |

## 8. Следующие шаги

После завершения этого плана и зелёного CI — отдельные планы для:
1. BUG-N11 (video aggregation, schema Chroma v2 + rollback-скрипт).
2. BUG-N20 (NLP parser через Ollama llama3, фиче-флаг).
3. PERF-N02 (батчированный InsightFace / multi-ONNX-session).
4. Prometheus-метрики + `/healthz`.
