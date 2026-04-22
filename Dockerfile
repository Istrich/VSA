# syntax=docker/dockerfile:1.7
# Vision Semantic Archive runtime image.
#
# Production target uses CUDA 12.1 runtime on Ubuntu 22.04 with ffmpeg and
# Python 3.11. Host the image on a CUDA-capable machine (RTX 3090 in the
# architecture doc) and expose the Streamlit port.

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY core ./core
COPY streamlit_app.py ./streamlit_app.py

EXPOSE 8501
ENV VSA_ALLOW_CPU=false

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
