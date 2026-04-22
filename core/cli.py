"""Command-line entry point for Vision Semantic Archive.

Usage::

    python -m core.cli index /path/to/media
    python -m core.cli doctor
    python -m core.cli download-models
    python -m core.cli ui   # equivalent to `streamlit run streamlit_app.py`

The CLI is intentionally thin — it delegates to the same ``ServiceContainer``
used by the Streamlit UI so both surfaces share the process-wide singletons.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .compatibility import run_compatibility_checks
from .config import get_settings
from .container import ServiceContainer
from .logging_config import configure_logging

LOGGER = logging.getLogger("vsa.cli")


def _cmd_index(args: argparse.Namespace) -> int:
    container = ServiceContainer.get()
    indexer = container.indexer()
    stats = asyncio.run(
        indexer.index_directory(
            root_directory=args.directory,
            keyframe_interval_sec=args.keyframe_interval,
            scene_delta_threshold=args.scene_delta,
        )
    )
    LOGGER.info("Indexing stats: %s", stats.model_dump())
    asyncio.run(indexer.ollama_client.aclose())
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    container = ServiceContainer.get()
    results = run_compatibility_checks(container=container)
    exit_code = 0
    for check in results:
        prefix = {"PASS": "[ ok ]", "WARN": "[warn]", "FAIL": "[fail]"}.get(
            check.status, "[ ?? ]"
        )
        sys.stdout.write(f"{prefix} {check.name}: {check.details}\n")
        if check.status == "FAIL":
            exit_code = 1
    return exit_code


def _cmd_download_models(_: argparse.Namespace) -> int:
    container = ServiceContainer.get()
    downloader = container.model_downloader
    missing = [
        name
        for name, status in downloader.get_status_map().items()
        if status != "FOUND"
    ]
    if not missing:
        LOGGER.info("All required model files are already present.")
        return 0
    for model_name in missing:
        LOGGER.info("Downloading %s...", model_name)
        downloader.download_model(model_name)
    LOGGER.info("Finished downloading %d model(s).", len(missing))
    return 0


def _cmd_ui(_: argparse.Namespace) -> int:
    import subprocess

    app_path = Path(__file__).resolve().parent.parent / "streamlit_app.py"
    result = subprocess.run(
        ["streamlit", "run", str(app_path)],
        check=False,
    )
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by ``python -m core.cli`` and the ``vsa`` script."""
    parser = argparse.ArgumentParser(prog="vsa", description="Vision Semantic Archive CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Index a media directory")
    index_parser.add_argument("directory")
    index_parser.add_argument("--keyframe-interval", type=int, default=2)
    index_parser.add_argument("--scene-delta", type=float, default=15.0)
    index_parser.set_defaults(func=_cmd_index)

    doctor_parser = sub.add_parser("doctor", help="Run compatibility checks")
    doctor_parser.set_defaults(func=_cmd_doctor)

    dm_parser = sub.add_parser("download-models", help="Download missing model assets")
    dm_parser.set_defaults(func=_cmd_download_models)

    ui_parser = sub.add_parser("ui", help="Launch the Streamlit UI")
    ui_parser.set_defaults(func=_cmd_ui)

    args = parser.parse_args(argv)
    configure_logging(get_settings().log_level)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
