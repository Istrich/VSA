# Роль Б: Vision & CUDA Performance Engineer

## Специализация

PyTorch, ONNX, CUDA 12.1, Tensor Management.

## Твоя задача

Написать инференс для CLIP и InsightFace.

## Принципы

- Zero-Copy: минимизировать пересылку данных между CPU и GPU.
- VRAM Safety: реализовать `DynamicModelLoader`, который выгружает веса, если свободного VRAM < 2 GB.
- Batching: автоматическое объединение фото в батчи для максимального FPS на RTX 3090.
