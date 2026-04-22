# Audit Bugfix Changelog

Date: 2026-04-22
Scenario: Багфикс

## Debugger (change-log)

- `core/vision.py`:
  - Убран hard-fail на import/инициализации без CUDA.
  - Добавлен lazy-init (`ensure_ready`) и CPU fallback через `VSA_ALLOW_CPU=1`.
  - Синхронизирован путь InsightFace через `INSIGHTFACE_HOME` (по умолчанию `./models/faces`).
  - Добавлено использование локального CLIP веса `./models/vision/open_clip_vit_l_14/model.safetensors` при наличии.
- `core/indexer.py`:
  - Исправлены silent-fail ветки индексатора: добавлено логирование исключений.
  - Улучшен `OllamaClient`: переиспользуемый `httpx.AsyncClient`, retry/backoff.
  - Дедупликация: при совпадении hash обновляется path (`rebind_path_by_hash`), вместо немого skip.
  - `media_id` стал детерминирован по hash; снижен риск hash/id рассинхронизации.
  - Выравнен дефолт `keyframe_interval_sec` до 2.
- `core/db.py`:
  - Включены `WAL` и `busy_timeout`.
  - `upsert_media` переведён на конфликт по `hash` для корректного поведения при перемещении файла.
  - Добавлена JSON-валидация `metadata_json` перед записью.
  - Добавлена санитизация FTS5-запроса для защиты от `OperationalError` на спецсимволах.
- `core/search.py`:
  - API поиска поддерживает `SearchQuery` (pydantic модель) для межмодульной передачи данных.
  - `search_with_uploaded_face` исправлен для кросс-платформенной работы temp-файлов.
- `streamlit_app.py`:
  - Добавлена вкладка `Index` для запуска индексирования из UI.
  - Добавлен видео-preview (`st.video`) со стартом по `best_frame_timestamp_sec`.
- `core/model_downloader.py`:
  - Переведён с `urlretrieve` на `httpx` streaming download.
  - Добавлены retry, timeout, атомарная запись через `.part`.
  - Добавлена поддержка zip-архивов (для `buffalo_l`) с распаковкой.
  - Поддержана проверка sha256 (если указана в спецификации).
- `core/models.py`:
  - Добавлены pydantic модели `SearchWeights`, `SearchResult`.
  - `ModelSpec` переведён в pydantic.
  - Обновлена спецификация `insightface_buffalo_l` на официальный `buffalo_l.zip` + ожидаемый путь внутри пакета.
- `core/compatibility.py`:
  - Расширены проверки: `ffprobe`, model files, ChromaDB version, PyTorch/CUDA runtime.
  - Смягчена жёсткая привязка ONNX Runtime только к 1.17.x.
- Добавлены: `.gitignore`, `pyproject.toml`, `requirements.txt`.

## Guardian

VERDICT: APPROVE

Причины:
- Закрыты критичные блокеры dev-runtime (GPU hard-fail, CPU fallback).
- Исправлены data-integrity и dedup edge-cases (hash conflict, path rebind).
- Снижен риск падений FTS5 от пользовательского ввода.
- Улучшена диагностируемость ошибок индексирования и Ollama запросов.

## Tester

Проверено:
- `python3 -m compileall core streamlit_app.py` — без синтаксических ошибок.
- IDE lints для изменённых файлов — без новых ошибок.

Ручной чек-лист:
1. Запустить UI: `streamlit run streamlit_app.py`.
2. Во вкладке `Settings/Status` прогнать compatibility checks.
3. Во вкладке `Index` проиндексировать папку с 1 изображением и 1 видео.
4. В `Search` проверить:
   - текстовый запрос со спецсимволами (`- : " AND`) не ломает FTS;
   - поиск по загруженному лицу работает;
   - для видео отображается preview со стартом по timestamp.

## Documenter

- Добавлен этот документ с полным change-log и тест-планом.
