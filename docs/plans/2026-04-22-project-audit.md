# Vision Semantic Archive — Project Audit (Vertical + Horizontal)

Date: 2026-04-22
Role: Analyzer / Architect (read-only audit, no code changes)
Scope: full repository (`core/*`, `streamlit_app.py`, `docs/*`)

## 1. Классификация задачи

Аудит + план доработок. Ближе всего к роли Analyzer с последующей эскалацией к Architect по подтверждённым находкам. Код не меняем, артефакт — этот файл.

## 2. Методология

- Вертикальный аудит: сквозной проход по E2E пути данных
  `Scanner → Vision (CLIP/Faces) → Caption (Ollama) → Storage (SQLite + Chroma) → Search (CLIP + Face + FTS) → UI`.
- Горизонтальный аудит: кросс-срезные концерны
  `deps · типы · ошибки · логирование · конфиг · тесты · CI · безопасность · производительность · наблюдаемость · документация · упаковка`.

---

## 3. Вертикальный аудит (по слоям)

### 3.1. Ingestion / Scanner (`core/indexer.py`)
- [BUG-01] `_index_image_file` и `_index_video_file` глушат любое исключение через `except Exception: return False`. Нет логирования, невозможно диагностировать причину. Нарушение guardrail «явная обработка ошибок, без голых except».
- [BUG-02] Нет UI‑точки запуска индексации (в `streamlit_app.py` только Search и Settings/Status). Индекс можно получить только вручную из Python REPL.
- [BUG-03] Дедупликация только по `hash`. При перемещении/переименовании файла хеш тот же → в `media.path` не обновится (хотя upsert по `id` есть, но `id` новый для каждого вызова `_build_media_id`). Риск UNIQUE на `hash` при повторной индексации.
- [BUG-04] `upsert_media` имеет `ON CONFLICT(id)`, но `hash` имеет `UNIQUE`. При новом `id` и том же `hash` — IntegrityError (перехватится BUG-01 и тихо будет `failed`).
- [BUG-05] Видео‑индексация **не** записывает CLIP/Face эмбеддинги для самой записи `video_media_id` (эмбеддинги есть только у кадров, путь кадра = путь видео). Это работает, но семантика «видео как сущность» размыта: одно видео может породить десятки кандидатов в Chroma с одинаковым `path`, конкурирующих за один слот кандидата в `HybridSearchEngine`.
- [BUG-06] `keyframe_interval_sec` в плане документа `= 2`, в коде `= 3`. Незначительный дрифт.
- [PERF-01] `InferenceService.get_faces` вызывается пер‑кадр без батчинга; CLIP батчится, InsightFace — нет.
- [PERF-02] `OllamaClient` создаёт новый `httpx.AsyncClient` на каждый запрос → нет connection‑pooling. На больших корпусах — заметный overhead.
- [PERF-03] `OLLAMA_NUM_PARALLEL` по умолчанию `1` → видео‑кадры фактически обрабатываются последовательно, даже при `asyncio.create_task`.
- [PERF-04] Хеш файла читается отдельным проходом (`_hash_file`) поверх CLIP (`PIL.open`) и Face (`cv2.imread`) — 3 чтения с диска. Для больших видео `_hash_file` полностью читает файл.
- [COMPAT-01] `tempfile.NamedTemporaryFile(delete=True)` с повторным открытием не работает на Windows (контекстно).

### 3.2. Vision (`core/vision.py`)
- [BUG-07] **Blocker на macOS/CPU‑dev**: `_assert_cuda_runtime()` при отсутствии CUDA или версии не `12.1.x` падает на импорте. Невозможно даже развернуть UI для проверки без GPU‑хоста. В репозитории видно `__pycache__/streamlit_app.cpython-314.pyc` — значит, попытки запуска были.
- [BUG-08] Singleton `InferenceService` через `__new__` + `__init__` инициализирует модели при импорте/первом обращении даже для задач, которым они не нужны (например, UI просмотра результатов).
- [BUG-09] `FaceAnalysis(name="buffalo_l")` ищет модели в `~/.insightface/models/buffalo_l/` (дефолт InsightFace), а `ModelRegistry` ожидает их в `./models/faces/buffalo_l/model.onnx`. Пути не синхронизированы.
- [BUG-10] `open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")` скачивает веса в свой кэш OpenCLIP. Файл `./models/vision/open_clip_vit_l_14/model.safetensors`, скачиваемый `ModelDownloader`, **не используется** → «сирота».
- [BUG-11] `model_downloader.py` для `insightface_buffalo_l` ссылается на **один** файл `w600k_r50.onnx`, а `buffalo_l` — это пак (`det_10g.onnx`, `2d106det.onnx`, `genderage.onnx`, `w600k_r50.onnx`, ...). Загрузка такой «подделанной» директории ломает `FaceAnalysis.prepare()`.
- [BUG-12] Нет CPU‑fallback / mock‑режима для разработки и unit‑тестов.

### 3.3. Ollama client (`core/indexer.py::OllamaClient`)
- [BUG-13] Ошибки Ollama (timeout, 5xx) маскируются `RuntimeError` и далее поглощаются общим `except Exception` индексера. Неинформативно.
- [BUG-14] Нет ретраев / backoff.
- [PERF-05] Семафор создан корректно, но без единого переиспользуемого клиента (см. PERF-02).

### 3.4. Storage (`core/db.py`)
- [BUG-15] `SQLiteMetadataDB._connect()` не включает WAL (`PRAGMA journal_mode=WAL;`) — при параллельной индексации и одновременном поиске возможны блокировки.
- [BUG-16] `search_captions` передаёт произвольный текст в `MATCH ?`. Пользовательский ввод со спецсимволами FTS5 (`-`, `"`, `:`, `*`, `^`, `AND/OR/NOT`) приведёт к `sqlite3.OperationalError`. Нужна санитизация/эскейпинг.
- [BUG-17] Хранение `metadata_json` строкой без JSON‑чека — возможна запись невалидного JSON через кастомный код (сейчас нет, но хрупко).
- [BUG-18] FTS‑триггеры `media_ai/ad/au` корректны для external‑content, но при массовой переиндексации цикл delete→insert на каждую строку дорог. Нет bulk‑варианта.
- [BUG-19] Нет миграций (Alembic/yoyo/ручной версия‑файл). Любое изменение схемы ломает существующие `./data/metadata.db`. Guardrail «не менять Chroma коллекции без плана миграции» формально соблюдён, а для SQLite не закреплён вовсе.

### 3.5. Search (`core/search.py`)
- [BUG-20] `HybridSearchEngine` при отсутствии одной из ветвей (например, пустой FTS) нормализует `clip_sim`/`face_sim` только по тем кандидатам, кто попал в `candidates`. Если кандидат пришёл только из FTS, у него `clip_sim=0`, `face_sim=0`, но после min‑max это станет `0` или `1.0` в вырожденном случае — перевес может быть нестабильным.
- [BUG-21] `_merge_fts_branch` использует `bm25` напрямую, а SQLite FTS5 возвращает значения, которые могут быть отрицательными; min‑max по набору ≠ 0 работает, но корректность интерпретации «меньше = лучше» не обозначена в `bm25(media_fts, ...)` (не задана весовая схема по столбцам — у нас один столбец, норм, но стоит оставить явно).
- [BUG-22] При нескольких лицах на референсе используется только `faces[0]`. Нет выбора лица пользователем, нет объединения (union по эмбеддингам).
- [BUG-23] NLP‑parser (Ollama `llama3` → JSON intents), заявленный в `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` §2 «Модуль C», **не реализован**.
- [BUG-24] Rerank заявлен по «Cosine + BM25», фактически это взвешенная сумма min‑max нормированных similarity. Расхождение с документацией.
- [BUG-25] `search_with_uploaded_face` использует `tempfile.NamedTemporaryFile(delete=True)` — см. COMPAT‑01.
- [TYPES-01] Публичный API `search(text_query, face_reference_path, top_k, weights)` не использует существующую модель `SearchQuery` (pydantic). Нарушение `.cursorrules` («Pydantic models for data passing between modules»).
- [TYPES-02] `SearchWeights` — dataclass, `ModelSpec` — dataclass, остальное — pydantic. Разнородность.

### 3.6. Compatibility (`core/compatibility.py`)
- [BUG-26] Жёсткая привязка `onnxruntime-gpu` к `1.17.x`. При обновлении CUDA/ORT тест будет ложно‑WARN.
- [BUG-27] Нет проверок:
  - наличия PyTorch/CUDA с нужной версией,
  - доступности моделей InsightFace/CLIP на диске (т.е. `ModelRegistry.is_ready()` не вызывается),
  - установленного ffprobe (не только ffmpeg),
  - ChromaDB версии.

### 3.7. UI (`streamlit_app.py`)
- [UX-01] Нет вкладки Index / Ingest.
- [UX-02] Нет превью видео, нет seek к `best_frame_timestamp_sec`.
- [UX-03] Downloader блокирует основной поток Streamlit. Для больших моделей зависает интерфейс.
- [UX-04] При нескольких лицах на референсе нет выбора.
- [UX-05] Результаты показывают путь, но не дают скопировать/открыть файл.
- [UX-06] Нет сохранения пресетов запросов и весов.
- [UX-07] `render_result_card` проверяет расширение в `{.jpg,.jpeg,.png,.webp,.bmp}` → для видео «Preview unavailable» без альтернативы (thumb по keyframe путь храним в SQLite, но не используем).

### 3.8. Model downloader (`core/model_downloader.py`)
- [SEC-01] `urlretrieve` без верификации контрольных сумм (sha256) и без принудительной TLS‑проверки (доверяем дефолту).
- [SEC-02] Нет ретраев, нет resume, нет таймаута.
- [BUG-28] см. BUG‑11 — некорректная URL для `buffalo_l`.

---

## 4. Горизонтальный аудит (кросс-срез)

### 4.1. Управление зависимостями
- [HZ-01] Отсутствует `requirements.txt` / `pyproject.toml`. Версии пакетов не фиксированы (streamlit, pydantic, chromadb, httpx, open_clip_torch, insightface, onnxruntime-gpu, torch, opencv-python, numpy, Pillow). Невоспроизводимая сборка.
- [HZ-02] Не задана версия Python (в `__pycache__` виден 3.14, архитектура декларирует 3.11+). Нет `python_requires` / `.python-version`.

### 4.2. Тесты / CI
- [HZ-03] Нет `tests/`, нет pytest‑конфигурации. Все «test reports» — ручная проверка. Нет unit/integration тестов для:
  - FTS delete/insert/update триггеров,
  - нормализации рангов,
  - keyframe‑экстрактора,
  - совместимости схемы,
  - poison‑query в FTS (BUG‑16),
  - deduplication.
- [HZ-04] Нет CI (GitHub Actions / другая).
- [HZ-05] Нет линтера/форматтера в конфиге (ruff/black/mypy) — рассыпанные type hints местами без проверки.

### 4.3. Конфигурация / окружение
- [HZ-06] Жёстко зашитые пути: `./data/metadata.db`, `./data/chroma`, `./models`, `http://localhost:11434`. Нет централизованного `Settings` (pydantic‑settings).
- [HZ-07] Нет `.env.example`. Единственная env‑переменная `OLLAMA_NUM_PARALLEL` не документирована.

### 4.4. Логирование / наблюдаемость
- [HZ-08] `logging` не используется ни в одном модуле. Все ошибки глотаются. Нет структурированных логов индексации.
- [HZ-09] Нет метрик (скорость индексации, размеры коллекций, VRAM). Нет `healthcheck`‑эндпоинта.
- [HZ-10] Нет progress‑bar для индексации в UI.

### 4.5. Безопасность
- [HZ-11] Нет валидации пользовательского ввода для FTS (BUG‑16).
- [HZ-12] Загрузка моделей без проверки контрольных сумм (SEC‑01).
- [HZ-13] Нет ограничения размера uploaded‑файла (Streamlit default 200MB, но можно понизить для face‑ref).

### 4.6. Документация / Проект
- [HZ-14] Нет `README.md` в корне (только архитектурный MD и план‑файлы).
- [HZ-15] Нет `.gitignore` (но `__pycache__` уже в репо‑дереве, `.git` существует — значит, риск коммита кэшей/БД/моделей).
- [HZ-16] Архитектурный документ расходится с кодом (Модуль C NLP Parser, Reranker Cosine+BM25, keyframe 2s).
- [HZ-17] Нет Makefile / taskfile, CLI‑команд для индексации, диагностики, миграций.

### 4.7. Упаковка / развёртывание
- [HZ-18] Нет Dockerfile / compose. Архитектурный документ декларирует RTX 3090, но нет рецепта развёртывания и pin'а CUDA‑образа.
- [HZ-19] Нет entry‑points (`vsa-index`, `vsa-ui`).
- [HZ-20] `core/__init__.py` пустой (нет публичного API).

### 4.8. Типизация / `.cursorrules`
- [HZ-21] `.cursorrules` требует pydantic для обмена между модулями. `HybridSearchEngine.search` принимает сырые параметры, а `SearchQuery` модель определена, но не используется. Также результаты поиска — `dict[str, Any]`, хотя можно ввести `SearchResult(BaseModel)`.
- [HZ-22] `ExtractedKeyframe` — frozen dataclass. Небольшое расхождение со стилем.

### 4.9. Совместимость с существующими guardrail'ами
- [OK-01] Chroma collection names (`embeddings_clip`, `embeddings_faces`) не меняются.
- [OK-02] Dependency Injection соблюдается (компоненты принимаются в конструкторах Indexer/Search).
- [WARN-01] Новые сущности (NLP parser, migrations) должны добавляться без слома DI — заложить в план.

---

## 5. Сводная таблица находок

| ID | Слой | Severity | Кратко |
|---|---|---|---|
| BUG-01 | Indexer | High | Глотание исключений без логов |
| BUG-02 | UI | High | Нет UI‑индексации |
| BUG-03/04 | Indexer/DB | High | Dedup по hash ломается при новых `id` |
| BUG-05 | Indexer | Medium | Видео‑сущность размыта в Chroma |
| BUG-07 | Vision | Critical (для dev) | Нет CPU‑fallback, падение на macOS |
| BUG-09 | Vision | High | Пути InsightFace ≠ `ModelRegistry` |
| BUG-10 | Downloader | High | CLIP safetensors не используется |
| BUG-11/28 | Downloader | Critical | Неверный URL для `buffalo_l` (пак→один файл) |
| BUG-13/14 | Ollama | Medium | Нет ретраев/диагностики |
| BUG-15 | DB | Medium | Нет WAL для SQLite |
| BUG-16 | DB | High | FTS5 инъекция ломает запрос |
| BUG-19 | DB | High | Нет миграций |
| BUG-20/21 | Search | Medium | Нормализация ранжирования неустойчива |
| BUG-23 | Search | Medium | NLP‑parser не реализован (док‑дрифт) |
| BUG-26/27 | Compat | Medium | Слабое покрытие проверок |
| UX-01..07 | UI | Medium | Отсутствуют ключевые UX‑функции |
| SEC-01/02 | Download | High | Нет checksums/таймаутов/TLS‑pin |
| HZ-01/02 | Packaging | Critical | Нет requirements/pyproject/python‑pin |
| HZ-03/04/05 | Quality | High | Нет тестов, CI, линтеров |
| HZ-06/07 | Config | Medium | Нет единого Settings, `.env.example` |
| HZ-08/09/10 | Observability | High | Нет логов, метрик, progress |
| HZ-14/15/16 | Docs | Medium | README/.gitignore/док‑дрифт |
| HZ-18/19 | Deploy | Medium | Нет Docker/CLI |
| HZ-21/22 | Types | Medium | Pydantic‑контракты нарушены |

---

## 6. Рекомендуемая дорожная карта (по приоритету)

### P0 — Blocker (чинится первым)
1. **HZ-01/02**: добавить `pyproject.toml` + `requirements.txt` с pin'ами версий (torch 2.2.x+cu121, open_clip_torch, insightface, onnxruntime-gpu 1.17.x, chromadb 0.4.x, streamlit≥1.32, httpx, pydantic≥2, opencv-python, numpy, Pillow). Зафиксировать Python 3.11.
2. **BUG-07/08/12**: убрать жёсткий assert CUDA 12.1 из `__init__` → превратить в `ensure_ready()` с env‑флагом `VSA_ALLOW_CPU=1` и lazy‑инициализацией. Дать честный mock/NoGPU‑путь для тестов и Streamlit preview.
3. **BUG-09/10/11/28**: синхронизировать пути моделей и загрузку:
   - CLIP: не тянуть safetensors отдельно, использовать кэш OpenCLIP **или** подгружать через `pretrained="./models/...`.
   - InsightFace: скачивать полный пак `buffalo_l.zip` с официального `onnx-models` bucket и распаковывать в `~/.insightface/models/buffalo_l/` (или проставлять `INSIGHTFACE_HOME=./models/faces`).
4. **HZ-15**: добавить `.gitignore` (`__pycache__/`, `data/`, `models/`, `*.db`, `*.sqlite*`, `.env`).

### P1 — High
5. **BUG-01/13/14** + **HZ-08**: ввести `logging` через `logging.getLogger(__name__)`, конфиг через `dictConfig`; заменить глушение на явные `logger.exception(...)` + типовые ошибки (`OllamaError`, `IngestError`). Добавить ретраи Ollama (tenacity/ручной backoff).
6. **BUG-16**: написать санитайзер FTS5‑запроса (эскейп кавычек, отбрасывание управляющих операторов, режим phrase) и покрыть тестами.
7. **BUG-19**: принять лёгкий механизм миграций (простые версии + `CREATE IF NOT EXISTS` + `schema_version` таблица) и зафиксировать в `docs/plans/`.
8. **HZ-03/04/05**: поднять `tests/` (pytest) + `ruff`+`mypy`+`black` в CI (GitHub Actions). Минимум:
   - unit: FTS5 triggers, санитайзер, rerank math, dedup,
   - integration: end‑to‑end на 1 изображении и 1 видео в CPU/mock‑режиме.
9. **BUG-03/04/05**: переделать dedup/upsert — ключом делать `hash` (или составной `hash+path`), а `id` выводить детерминировано из `hash`. Для видео — хранить сущность `video` (с агрегированным caption) и сущности `frame` со ссылкой на родителя.
10. **SEC-01/02**: добавить sha256‑хеши к `ModelSpec`, проверять после загрузки; выставить таймауты, ретраи, атомарную запись через `.part`→rename.
11. **BUG-15**: включить WAL и `busy_timeout` для SQLite.

### P2 — Medium
12. **UX-01/03**: добавить вкладку Index в Streamlit, запускать `asyncio.run(indexer.index_directory(...))` в background thread, показывать progress + счётчики.
13. **UX-02/07**: превью видео через `st.video(path, start_time=best_frame_timestamp_sec)`.
14. **UX-04**: карусель/выбор лица при множественном детекте в референсе.
15. **BUG-20/21/24**: зафиксировать формулу ранжирования (нормализация учитывает все кандидаты, не только тех, кто попал в обе ветви; BM25‑веса по столбцам; документная синхронизация).
16. **BUG-23**: реализовать NLP parser как отдельный модуль `core/nlp.py` (опциональный, за фичефлагом), чтобы не ломать guardrail DI.
17. **HZ-06/07**: `pydantic-settings` + `.env.example` (пути, Ollama URL, параллелизм, модели).
18. **HZ-21/22**: унифицировать обмен — ввести `SearchResult(BaseModel)`, `IndexingStats(BaseModel)`, использовать `SearchQuery` в `HybridSearchEngine.search`.
19. **PERF-01/02**: переиспользуемый `httpx.AsyncClient`, батчинг InsightFace (batched `app.get` или многопоточность по ONNX сессиям).

### P3 — Low / Nice-to-have
20. **HZ-18/19**: Dockerfile (nvidia/cuda:12.1-runtime) + `compose.yml` + CLI `vsa index ./folder`, `vsa ui`.
21. **HZ-14/16**: обновить `README.md`, свести к реальности архитектурный документ (keyframe interval, реализованный reranker, статус NLP parser).
22. **BUG-26/27**: расширить compatibility checks (Chroma version, torch+cuda, model files via `ModelRegistry.is_ready()`).
23. **HZ-09**: простые метрики (prometheus_client) и endpoint `/metrics`.

---

## 7. Риски и rollback
- Любая смена схемы SQLite/Chroma — обязательна миграция + rollback‑скрипт (drop table / rename). Зафиксировать уже сейчас, до первых серьёзных рефакторингов.
- Изменение URL загрузки моделей — риск ссылок на нестабильные зеркала. Предпочитать официальные Hugging Face репозитории + sha256.
- Отказ от CUDA‑ассерта — риск тихой работы на CPU с unacceptable latency. Mitigation: явное логирование «CPU mode engaged», UI‑бейдж.

## 8. Следующие шаги
- Если пункты P0 подтверждаются — завести по каждому отдельный план‑файл `docs/plans/YYYY-MM-DD-<slug>.md` и перейти в фазу Implementer.
- Для P1 (dedup, миграции) — предварительно пересмотреть схему (Architect).
- Для P2/P3 — группировать в минорные итерации.
