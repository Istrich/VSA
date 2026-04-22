# Архитектура сервиса: Vision Semantic Archive (VSA)

## Роли агентов (System Prompts)

Вынесены в отдельные файлы:

- `docs/system-prompts/role-a-database-schema-architect.md`
- `docs/system-prompts/role-b-vision-cuda-performance-engineer.md`
- `docs/system-prompts/role-v-llm-vlm-orchestrator.md`
- `docs/system-prompts/role-g-search-ui-integrator.md`

## 1. Технологический стек и версии

- **Язык:** Python 3.11+
- **UI/Frontend:** Streamlit 1.32+ (максимальная скорость сборки интерфейса)
- **Vector DB:** ChromaDB 0.4.x (локальное хранилище, не требует Docker)
- **Metadata/FTS:** SQLite 3.x (с поддержкой FTS5 для полнотекстового поиска)
- **Deep Learning Framework:** PyTorch 2.2+ (CUDA 12.1)

### Inference Engines

- **Faces:** InsightFace (модель `buffalo_l`) через `onnxruntime-gpu`
- **Semantic (CLIP):** `open_clip_torch` (модель `ViT-L-14 / openai`)
- **VLM/LLM:** Ollama API (модели `moondream2` для описаний и `llama3` для парсинга запросов)

## 2. Компоненты системы

### Модуль A: Ingestion Pipeline (фоновая индексация)

- **Scanner:** рекурсивный обход директорий, хеширование файлов (`SHA-256`) для предотвращения дублей
- **Vision Worker (мультипоточность):**
  - **Stage 1 (Faces):** извлечение эмбеддингов лиц (512-d вектор)
  - **Stage 2 (CLIP):** извлечение семантического вектора кадра (768-d вектор)
  - **Stage 3 (VLM):** запрос к Ollama (`moondream2`) для генерации короткого текстового описания (caption)
- **Video Handler:** нарезка видео через FFmpeg (1 кадр в 2 сек) -> обработка кадров как фото -> агрегация описаний через LLM

### Модуль B: Storage Layer (гибридная БД)

#### ChromaDB Collections

- `collection_clip`: `{id: file_path, vector: clip_emb, metadata: {type: image/video}}`
- `collection_faces`: `{id: face_id, vector: face_emb, metadata: {file_id: path}}`

#### SQLite (FTS5)

- Таблица `media`: `file_path`, `caption_text`, `created_at`, `is_video`

### Модуль C: Search Engine (интеллектуальный поиск)

- **NLP Parser:** преобразует запрос пользователя через Ollama (`llama3`) в JSON-структуру:

```json
{
  "search_type": "hybrid",
  "person_ref": true,
  "text_description": "big tree near water",
  "location": "Japan"
}
```

- **Hybrid Retriever:**
  - если `person_ref = true`: сначала поиск по `collection_faces`
  - параллельно: семантический поиск по `collection_clip` + полнотекстовый поиск по SQLite (FTS5)
- **Reranker:** сортировка результатов по совокупному весу (Cosine Similarity + BM25 Score)

## 3. Схема взаимодействия компонентов

```mermaid
flowchart TD
    A[Файловая система] --> B[Scanner]
    B --> C[Vision Worker]
    C --> C1[Stage 1: Faces]
    C --> C2[Stage 2: CLIP]
    C --> C3[Stage 3: VLM Caption]
    A --> V[Video Handler + FFmpeg]
    V --> C

    C1 --> D1[(ChromaDB: collection_faces)]
    C2 --> D2[(ChromaDB: collection_clip)]
    C3 --> D3[(SQLite FTS5: media)]
    V --> D3

    U[Пользовательский запрос] --> P[NLP Parser (Ollama llama3)]
    P --> H[Hybrid Retriever]
    H --> D1
    H --> D2
    H --> D3
    H --> R[Reranker]
    R --> O[Результаты поиска]
```

## 4. Оптимизация под RTX 3090 (24GB VRAM)

| Задача | Ресурс VRAM | Технология |
|---|---:|---|
| CLIP (ViT-L-14) | ~2.5 GB | Постоянно в памяти (Fast inference) |
| InsightFace (ONNX) | ~1.0 GB | На CUDA-провайдере |
| Ollama (Llava/Llama3) | ~8-12 GB | Динамическое управление через Ollama |
| Система/Overhead | ~2.0 GB | Резерв |
| **Итого** | **~15-18 GB** | **Запас 6-9 GB для стабильной работы** |
