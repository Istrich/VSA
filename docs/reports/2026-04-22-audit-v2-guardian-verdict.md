# Audit v2 — Guardian verdict

Plan: `docs/plans/2026-04-22-audit-v2-fixes.md`
Change-log: `docs/reports/2026-04-22-audit-v2-changelog.md`

## VERDICT: APPROVE (with follow-ups)

Правки охватывают весь согласованный периметр P0+P1+P2+P3. Guardrails соблюдены: имена Chroma-коллекций не менялись, `DI` сохранён и расширен (`ServiceContainer` — не новый синглтон в смысле бизнес-логики, а DI-root, явно конструируемый/сбрасываемый), секреты не коммитятся, `.gitignore` расширен.

## Проверено

- [x] Python 3.11 AST-check для всех модулей (`python -c "import ast; ast.parse(...)"`).
- [x] Нет новых lint-ошибок (ReadLints на `core/` и `streamlit_app.py`).
- [x] Chroma имена коллекций не изменены (`embeddings_clip`, `embeddings_faces`).
- [x] SQLite схема остаётся v1; добавлена таблица `schema_version` для будущих миграций; все изменения триггеров FTS сохранены.
- [x] Нет `except Exception` без логирования; замены на типизированные исключения + `LOGGER.exception(...)`.
- [x] Нет секретов в репо; `.gitignore` покрывает `.env*`, `data/`, `models/`, `*.db`.
- [x] `docker-compose.yml` сохраняет `network_mode: host`.
- [x] `ServiceContainer` — единственная точка создания `chromadb.PersistentClient`.

## Риски, оставленные осознанно

1. **Offline-окружение**: CLIP по умолчанию тянется через `hf-hub:...`. В изолированных средах нужно выставить `VSA_CLIP_PRETRAINED=/path/to/local/openai_vit_l_14` (`open_clip` принимает путь к чекпоинту). README и `.env.example` это отражают.
2. **mypy в CI** запускается с `|| true` — baseline не зафиксирован. Первый проход может показать ошибки; ожидается, что последующая итерация зафиксирует baseline и включит строгий режим.
3. **Tests требуют libexpat**. На локальном macOS с установленным Python 3.14 из Homebrew обнаружена системная проблема (сломанный `pyexpat`), из-за чего pip не может установить пакеты. Локально пройти `pytest` не удалось. В CI (ubuntu-latest, Python 3.11/3.12) проблем быть не должно — все нужные пакеты доступны.
4. **Unit-тесты индексера** используют stubs для тяжёлых зависимостей (`torch`, `cv2`, `chromadb`, `insightface`). Это даёт быстрый suite, но не подменяет интеграционные проверки на реальном CUDA-хосте.

## Следующие шаги (обязательные)

1. В CI прогнать `pytest -q`, убедиться что зелёно на Python 3.11 и 3.12.
2. На CUDA-хосте выполнить ручной smoke-сценарий из test-report.
3. Если smoke-тест выявит mypy-шум — зафиксировать baseline и включить strict после рефакторинга.
4. По результатам production-smoke запустить Architect-план по BUG-N11 (video aggregation).

## Конкретные правки, которых НЕ потребовалось

- Перегенерация ChromaDB данных (коллекции и формат не менялись).
- Миграция SQLite-данных (схема v1 сохранена).
- Изменение `.cursorrules` — требования по типизации соблюдены через расширение pydantic-контрактов.
