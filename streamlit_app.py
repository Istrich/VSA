"""Streamlit UI for VSA hybrid search.

All services (ChromaDB, SQLite, InferenceService, OllamaClient) come from the
process-wide ``ServiceContainer`` so Streamlit's tab switching no longer
triggers a duplicate ``PersistentClient`` (BUG-N01). Indexing runs in a
background thread with live progress and cooperative cancellation (BUG-N02).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from core.compatibility import run_compatibility_checks
from core.config import get_settings
from core.container import ServiceContainer
from core.logging_config import configure_logging
from core.models import IndexingStats, SearchResult, SearchWeights


configure_logging(get_settings().log_level)


@st.cache_resource(show_spinner=False)
def get_container() -> ServiceContainer:
    """Build or reuse the single process-wide service container."""
    return ServiceContainer.get()


def _render_result_card(result: SearchResult) -> None:
    """Render one search result with preview and metadata."""
    path = result.path
    path_obj = Path(path)
    caption = result.caption or "No caption available"

    with st.container(border=True):
        suffix = path_obj.suffix.lower()
        if path_obj.exists() and suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            st.image(str(path_obj), use_container_width=True)
        elif path_obj.exists() and suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            start_time = result.best_frame_timestamp_sec or 0.0
            st.video(str(path_obj), start_time=float(start_time))
        else:
            st.markdown("**Preview unavailable**")

        st.markdown(f"**Path:** `{path}`")
        st.markdown(f"**Caption:** {caption}")
        st.markdown(
            f"**Score:** `{result.score:.4f}` | "
            f"CLIP: `{result.clip_sim:.4f}` | "
            f"Face: `{result.face_sim:.4f}` | "
            f"FTS: `{result.fts_score:.4f}`"
        )
        ts = result.best_frame_timestamp_sec
        if ts is not None:
            st.caption(f"Best match timestamp: {float(ts):.2f}s")
        if result.metadata:
            st.json(result.metadata, expanded=False)
        st.code(path, language=None)


def _run_index_thread(
    root_dir: str,
    keyframe_interval_sec: int,
    scene_delta_threshold: float,
    cancel_event: threading.Event,
    holder: dict[str, Any],
) -> None:
    """Worker executed in a background thread for media indexing."""
    container = ServiceContainer.get()
    indexer = container.indexer()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_cancel = asyncio.Event()

    def mirror_cancel() -> None:
        while not cancel_event.is_set() and not holder["done"]:
            time.sleep(0.2)
        if not holder["done"]:
            loop.call_soon_threadsafe(async_cancel.set)

    mirror = threading.Thread(target=mirror_cancel, daemon=True)
    mirror.start()

    def on_progress(stats: IndexingStats, media_path: Path) -> None:
        holder["stats"] = stats.model_dump()
        holder["current"] = str(media_path)

    try:
        stats = loop.run_until_complete(
            indexer.index_directory(
                root_directory=root_dir,
                keyframe_interval_sec=keyframe_interval_sec,
                scene_delta_threshold=scene_delta_threshold,
                cancel_event=async_cancel,
                progress_callback=on_progress,
            )
        )
        holder["stats"] = stats.model_dump()
    except Exception as exc:
        holder["error"] = str(exc)
    finally:
        try:
            loop.run_until_complete(indexer.ollama_client.aclose())
        except Exception:
            pass
        loop.close()
        holder["done"] = True


def _render_index_tab() -> None:
    """Render indexing workflow with progress and cancellation."""
    st.subheader("Index Media Directory")
    container = get_container()

    if not container.model_registry.is_ready():
        missing = [
            name
            for name, status in container.model_registry.get_model_status().items()
            if status != "FOUND"
        ]
        st.warning(
            "Some model files are missing: "
            + ", ".join(missing)
            + ". Open the Settings/Status tab to download them before indexing."
        )

    root_dir = st.text_input("Directory path", value=".")
    keyframe_interval = st.number_input("Keyframe interval (sec)", min_value=1, value=2, step=1)
    scene_delta = st.slider("Scene delta threshold", min_value=1.0, max_value=50.0, value=15.0)

    state = st.session_state
    state.setdefault("index_thread", None)
    state.setdefault("index_cancel_event", None)
    state.setdefault("index_holder", None)

    col_run, col_cancel = st.columns(2)
    with col_run:
        run_clicked = st.button("Run Indexing", type="primary", disabled=state["index_thread"] is not None)
    with col_cancel:
        cancel_clicked = st.button("Cancel", disabled=state["index_thread"] is None)

    if run_clicked and state["index_thread"] is None:
        cancel_event = threading.Event()
        holder: dict[str, Any] = {
            "stats": IndexingStats().model_dump(),
            "current": "",
            "done": False,
            "error": None,
        }
        thread = threading.Thread(
            target=_run_index_thread,
            args=(
                root_dir,
                int(keyframe_interval),
                float(scene_delta),
                cancel_event,
                holder,
            ),
            daemon=True,
        )
        thread.start()
        state["index_thread"] = thread
        state["index_cancel_event"] = cancel_event
        state["index_holder"] = holder

    if cancel_clicked and state["index_cancel_event"] is not None:
        state["index_cancel_event"].set()

    thread = state["index_thread"]
    holder = state["index_holder"]
    if thread is not None and holder is not None:
        progress_area = st.empty()
        status_area = st.empty()
        while thread.is_alive():
            stats = holder.get("stats") or {}
            total = max(stats.get("total_candidates", 0) or 0, 1)
            done = (
                stats.get("indexed", 0)
                + stats.get("skipped", 0)
                + stats.get("failed", 0)
            )
            progress_area.progress(min(1.0, done / total))
            status_area.markdown(
                f"Indexed: {stats.get('indexed', 0)} | "
                f"Skipped: {stats.get('skipped', 0)} | "
                f"Failed: {stats.get('failed', 0)} | "
                f"Total: {stats.get('total_candidates', 0)} | "
                f"Current: `{holder.get('current', '')}`"
            )
            time.sleep(0.5)
        thread.join()
        stats = holder.get("stats") or {}
        progress_area.progress(1.0)
        if holder.get("error"):
            st.error(f"Indexing failed: {holder['error']}")
        else:
            st.success(
                f"Done. Indexed: {stats.get('indexed', 0)} | "
                f"Skipped: {stats.get('skipped', 0)} | "
                f"Failed: {stats.get('failed', 0)} of "
                f"{stats.get('total_candidates', 0)}."
            )
        state["index_thread"] = None
        state["index_cancel_event"] = None


def _render_search_tab() -> None:
    """Render hybrid search workflow."""
    with st.sidebar:
        st.subheader("Search Settings")
        top_k = st.slider("Top-K results", min_value=1, max_value=100, value=20, step=1)
        w_clip = st.slider("Weight: CLIP", 0.0, 1.0, 0.4, 0.05)
        w_face = st.slider("Weight: Face", 0.0, 1.0, 0.6, 0.05)
        w_fts = st.slider("Weight: FTS", 0.0, 1.0, 0.0, 0.05)

    weights = SearchWeights(clip=w_clip, face=w_face, fts=w_fts)
    engine = get_container().search_engine()

    text_query = st.text_input("Text query", placeholder="e.g. near a big tree at sunset")
    settings = get_settings()
    uploaded = st.file_uploader(
        "Face reference image (optional)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )

    if st.button("Run Search", type="primary"):
        if not text_query.strip() and uploaded is None:
            st.warning("Provide text query and/or face reference image.")
            return

        uploaded_bytes: bytes | None = None
        uploaded_suffix = ".jpg"
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            size_mb = len(uploaded_bytes) / (1024 * 1024)
            if size_mb > settings.max_upload_size_mb:
                st.error(
                    f"Uploaded file is {size_mb:.1f} MB, which exceeds the "
                    f"{settings.max_upload_size_mb} MB limit. "
                    "Adjust VSA_MAX_UPLOAD_SIZE_MB to override."
                )
                return
            uploaded_suffix = Path(uploaded.name).suffix or ".jpg"

        with st.spinner("Running hybrid search..."):
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
                _render_result_card(result)


def _render_settings_status_tab() -> None:
    """Render model status and compatibility diagnostics."""
    container = get_container()
    registry = container.model_registry
    downloader = container.model_downloader

    st.subheader("Model Status")
    st.caption(
        "CLIP weights are fetched automatically by open_clip (HuggingFace Hub). "
        "InsightFace buffalo_l pack must be available on disk."
    )
    status_map = registry.get_model_status()
    specs = registry.get_specs()

    for spec in specs:
        status = status_map.get(spec.name, "MISSING")
        row_left, row_mid, row_right = st.columns([3, 2, 2])
        with row_left:
            st.markdown(f"**{spec.name}**")
            st.caption(f"`{registry.root / spec.relative_path}`")
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
    st.caption("Runtime checks for storage, Ollama, InsightFace, FFmpeg, CUDA providers.")
    if st.button("Run Compatibility Checks"):
        checks = run_compatibility_checks(container=container)
        for check in checks:
            if check.status == "PASS":
                st.success(f"{check.name}: {check.details}")
            elif check.status == "WARN":
                st.warning(f"{check.name}: {check.details}")
            else:
                st.error(f"{check.name}: {check.details}")


def main() -> None:
    """Streamlit app entrypoint."""
    settings = get_settings()
    st.set_page_config(page_title="VSA Hybrid Search", page_icon=None, layout="wide")
    st.title("Vision Semantic Archive")
    st.caption(
        f"Hybrid Search: text query + optional face reference. "
        f"Data: `{settings.data_dir}` | Models: `{settings.models_dir}`"
    )

    tab_search, tab_index, tab_settings = st.tabs(["Search", "Index", "Settings/Status"])
    with tab_search:
        _render_search_tab()
    with tab_index:
        _render_index_tab()
    with tab_settings:
        _render_settings_status_tab()


if __name__ == "__main__":
    main()
