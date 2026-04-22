"""Streamlit UI for VSA hybrid search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from core.compatibility import run_compatibility_checks
from core.model_downloader import ModelDownloader
from core.models import ModelRegistry
from core.search import HybridSearchEngine, SearchWeights


@st.cache_resource(show_spinner=False)
def get_search_engine() -> HybridSearchEngine:
    """Create singleton-like search engine for Streamlit session."""
    return HybridSearchEngine()


@st.cache_resource(show_spinner=False)
def get_model_registry() -> ModelRegistry:
    """Create model registry for settings/status view."""
    return ModelRegistry()


@st.cache_resource(show_spinner=False)
def get_model_downloader() -> ModelDownloader:
    """Create model downloader for settings/status actions."""
    return ModelDownloader()


def render_result_card(result: dict[str, Any]) -> None:
    """Render one search result with preview and metadata."""
    path = result.get("path", "")
    path_obj = Path(path)
    metadata = result.get("metadata", {})
    caption = result.get("caption") or "No caption available"

    with st.container(border=True):
        if path_obj.exists() and path_obj.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            st.image(str(path_obj), use_container_width=True)
        else:
            st.markdown("**Preview unavailable**")

        st.markdown(f"**Path:** `{path}`")
        st.markdown(f"**Caption:** {caption}")
        st.markdown(
            (
                f"**Score:** `{result.get('score', 0.0):.4f}` | "
                f"CLIP: `{result.get('clip_sim', 0.0):.4f}` | "
                f"Face: `{result.get('face_sim', 0.0):.4f}` | "
                f"FTS: `{result.get('fts_score', 0.0):.4f}`"
            )
        )
        if isinstance(metadata, dict) and metadata:
            ts = metadata.get("best_frame_timestamp_sec")
            if isinstance(ts, (float, int)):
                st.caption(f"Best match timestamp: {float(ts):.2f}s")
            st.json(metadata, expanded=False)


def render_search_tab() -> None:
    """Render hybrid search workflow."""
    with st.sidebar:
        st.subheader("Search Settings")
        top_k = st.slider("Top-K results", min_value=1, max_value=100, value=20, step=1)
        w_clip = st.slider("Weight: CLIP", 0.0, 1.0, 0.4, 0.05)
        w_face = st.slider("Weight: Face", 0.0, 1.0, 0.6, 0.05)
        w_fts = st.slider("Weight: FTS", 0.0, 1.0, 0.0, 0.05)

    weights = SearchWeights(clip=w_clip, face=w_face, fts=w_fts)
    engine = get_search_engine()

    text_query = st.text_input("Text query", placeholder="e.g. near a big tree at sunset")
    uploaded = st.file_uploader(
        "Face reference image (optional)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )

    if st.button("Run Search", type="primary"):
        if not text_query.strip() and uploaded is None:
            st.warning("Provide text query and/or face reference image.")
            return

        with st.spinner("Running hybrid search..."):
            uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
            uploaded_suffix = Path(uploaded.name).suffix if uploaded is not None else ".jpg"
            results = engine.search_with_uploaded_face(
                text_query=text_query,
                uploaded_bytes=uploaded_bytes,
                uploaded_suffix=uploaded_suffix,
                top_k=top_k,
                weights=weights,
            )

        if not results:
            st.info("No results found.")
            return

        st.success(f"Found {len(results)} results")
        columns = st.columns(3)
        for idx, result in enumerate(results):
            with columns[idx % 3]:
                render_result_card(result)


def render_settings_status_tab() -> None:
    """Render model status and compatibility diagnostics."""
    registry = get_model_registry()
    downloader = get_model_downloader()

    st.subheader("Model Status")
    status_map = registry.get_model_status()
    specs = registry.get_specs()

    for spec in specs:
        status = status_map.get(spec.name, "MISSING")
        row_left, row_mid, row_right = st.columns([3, 2, 2])
        with row_left:
            st.markdown(f"**{spec.name}**")
            st.caption(f"`./models/{spec.relative_path}`")
        with row_mid:
            if status == "FOUND":
                st.success("FOUND")
            else:
                st.error("MISSING")
        with row_right:
            if status == "MISSING":
                button_key = f"download_{spec.name}"
                if st.button("Download", key=button_key):
                    progress = st.progress(0, text=f"Downloading {spec.name}...")
                    try:
                        downloader.download_model(
                            model_name=spec.name,
                            progress_callback=lambda value: progress.progress(
                                int(value * 100),
                                text=f"Downloading {spec.name}... {int(value * 100)}%",
                            ),
                        )
                        progress.progress(100, text=f"{spec.name} downloaded")
                        st.success(f"Downloaded: {spec.name}")
                        st.rerun()
                    except Exception as exc:
                        progress.empty()
                        st.error(f"Failed to download {spec.name}: {exc}")
            else:
                st.caption("Ready")

    st.divider()
    st.subheader("Compatibility Checklist")
    st.caption("Runtime checks for storage, Ollama, InsightFace dependency, and FFmpeg.")
    if st.button("Run Compatibility Checks"):
        checks = run_compatibility_checks()
        for check in checks:
            if check.status == "PASS":
                st.success(f"{check.name}: {check.details}")
            elif check.status == "WARN":
                st.warning(f"{check.name}: {check.details}")
            else:
                st.error(f"{check.name}: {check.details}")


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="VSA Hybrid Search", page_icon="🔎", layout="wide")
    st.title("Vision Semantic Archive")
    st.caption("Hybrid Search: text query + optional face reference")

    tab_search, tab_settings = st.tabs(["Search", "Settings/Status"])
    with tab_search:
        render_search_tab()
    with tab_settings:
        render_settings_status_tab()


if __name__ == "__main__":
    main()

