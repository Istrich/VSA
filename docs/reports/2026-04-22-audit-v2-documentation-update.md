# Audit v2 — Documenter update

## Обновлённые документы

| Файл | Изменение |
|---|---|
| `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` | Синхронизирован с кодом v2: Settings/container/CLIP-hf-hub/InsightFace-layout/per-branch normalization. Расхождения §2 (NLP parser, rerank формула) переведены в §7 future work. |
| `README.md` | Новый: quickstart, layout, конфиг, workflow. |
| `.env.example` | Новый: полный список VSA_* с пояснениями. |
| `docs/plans/2026-04-22-audit-v2-fixes.md` | Новый план Architect для этой итерации. |
| `docs/reports/2026-04-22-audit-v2-changelog.md` | Change-log Implementer. |
| `docs/reports/2026-04-22-audit-v2-guardian-verdict.md` | APPROVE с follow-ups. |
| `docs/reports/2026-04-22-audit-v2-test-report.md` | Результаты Tester + ручной smoke-чек-лист. |

## Что теперь отражает действительность

- В `VISION_SEMANTIC_ARCHIVE_ARCHITECTURE.md` указано, что CLIP-веса тянет сам `open_clip` (HF-hub), а не отдельный safetensors.
- InsightFace path синхронизирован: `${VSA_INSIGHTFACE_HOME}/models/buffalo_l/`.
- Rerank описан как per-branch min-max + weighted sum; упоминание «Cosine + BM25 Score» убрано как устаревшее.
- NLP parser запросов и video-aggregation явно помечены как future work.

## Что осталось в очереди (future work)

1. План для BUG-N11 (Chroma v2: агрегированная сущность «видео», миграция, rollback).
2. План для BUG-N20 (NLP parser через Ollama `llama3` за фиче-флагом).
3. Plan для мониторинга/метрик (`/healthz`, Prometheus).
4. Включить строгий mypy в CI после фиксации baseline.
