# Audit v2 — Tester report

Plan: `docs/plans/2026-04-22-audit-v2-fixes.md`
Change-log: `docs/reports/2026-04-22-audit-v2-changelog.md`

## Среда тестирования

- macOS 25 (darwin arm64)
- Системный Python 3.14.4 (Homebrew) — обнаружен дефект в нативном
  `pyexpat` (несовпадение символа `_XML_SetAllocTrackerActivationThreshold`
  между Python-shipped `pyexpat.so` и системной `libexpat.1.dylib`), из-за
  чего `python3 -m pip install` невозможен. Локальный прогон `pytest`
  оказался недоступен; автоматические проверки перенесены в CI.
- GPU недоступен; проверки `InferenceService` / компат-чеки CUDA выполняются
  на production-хосте отдельно.

## Автоматические проверки (локально доступные)

| Проверка | Команда | Результат |
|---|---|---|
| Синтаксис Python 3.11-совместим (AST) | `python3 -c "import ast; ast.parse(...)"` на 21 `.py` файл | PASS (0 SyntaxError) |
| Отсутствие lint-ошибок | `ReadLints core/ streamlit_app.py tests/` | PASS (0 ошибок) |
| Закрытие BUG-N12 (f-string 3.11) | Ревью `SQLiteMetadataDB._sanitize_fts_query` — бэкслешей в expression нет | PASS |
| Удаление `__pycache__` из репо | `ls __pycache__` | PASS (каталог отсутствует) |
| `.gitignore` покрывает артефакты | Ревью | PASS |
| Chroma-коллекции не переименованы | diff `db.py` | PASS (`embeddings_clip`, `embeddings_faces`) |

## Покрытие unit-тестами (ожидаемо зелёное в CI)

| Файл | Что проверяется | Ключевые кейсы |
|---|---|---|
| `tests/test_fts_sanitizer.py` | BUG-N12 + защита FTS5 | unicode-токены, эскейп кавычек, «ядовитые» запросы, сквозной тест через реальный SQLite |
| `tests/test_db_dedup.py` | Дедуп/rebind + FTS-триггеры | `media_exists_by_hash`, `rebind_path_by_hash`, update-triggers |
| `tests/test_search_rerank.py` | BUG-N17/N18 | single-candidate → score=1.0, single-branch candidate сохраняет вклад FTS |
| `tests/test_model_downloader.py` | SEC-N02, BUG-N24, ретраи | SHA-mismatch → `ModelAssetError`, zip-пак распакован по InsightFace-layout, abs-path zip отвергается |
| `tests/test_config.py` | pydantic-settings | env-override, производные пути, валидация |
| `tests/test_indexer_async.py` | BUG-N08/N09/N10 | first-pass/second-pass skip, move → Chroma rebind, fake `InferenceService`/`OllamaClient` |

## Ручной smoke-чек-лист (для CUDA-хоста и/или CPU-fallback)

1. **Setup**
   ```bash
   cp .env.example .env
   pip install -e .[dev]
   python -m core.cli doctor
   ```
   Ожидание: список PASS/WARN/FAIL, на dev-ноутбуке `CUDA` = WARN, остальное — PASS.
2. **Models**
   ```bash
   python -m core.cli download-models
   ls models/insightface_home/models/buffalo_l/
   ```
   Ожидание: пак `buffalo_l` скачан и разложен по пути, который читает `FaceAnalysis(root=..., name="buffalo_l")`.
3. **Streamlit UI**
   ```bash
   streamlit run streamlit_app.py
   ```
   - открыть Search, Index, Settings/Status поочерёдно → в stderr **не** должно появиться «An instance of Chroma already exists…» (закрытие BUG-N01);
   - нажать Run Compatibility Checks → список чеков выводится без краша;
   - перейти на Index, ввести директорию с 1 jpg + 1 mp4 → запустить Run Indexing, увидеть progress-bar, счётчики, кнопку Cancel;
   - прервать индексацию → «Done. Indexed/Skipped/Failed» отражает частичный прогон.
4. **Search correctness**
   После индексации одного изображения и одного видео:
   - text query типа «tree» — должны вернуться результаты с ненулевым `clip_sim` и (если captioning Ollama поднят) `fts_score`;
   - загрузить face-reference → получить результаты с `face_sim > 0`;
   - проверить, что карточка видео воспроизводится с `start_time = best_frame_timestamp_sec` (BUG-N19).
5. **Path rebind (BUG-N10)**
   - Проиндексировать файл, переместить его, переиндексировать каталог. В логе (уровень INFO) должен появиться «Rebound N rows» + последующий поиск возвращает новый путь.

## CI-матрица (ожидаемый прогон)

GitHub Actions `ci.yml` запускает:
- `ruff check .` — должен быть зелёным. Настроенные правила `E, F, W, I, UP, B, SIM, PL` без исключённых категорий кроме указанных `PLR*`.
- `black --check .` — формат соблюдён.
- `mypy core` — с `|| true`, чтобы не блокировать первый прогон; ошибки фиксируются в следующей итерации.
- `pytest -q` — подключает `tests/conftest.py::_bootstrap_heavy_stubs` до импорта `core/*`, поэтому torch/onnxruntime/cv2/insightface/chromadb не требуются.

## Известные ограничения

- Локально pytest не прогонялся (см. «Среда тестирования»). В CI этот прогон обязателен перед merge.
- Интеграционные тесты на реальном GPU (CLIP ViT-L-14 + InsightFace buffalo_l + Ollama) выполняются вручную на production-хосте.
- `test_indexer_async.py::test_move_triggers_chroma_rebind` использует in-memory Fake Chroma; реальное поведение ChromaDB при `update(where=...)` проверяется ручным сценарием §5.
