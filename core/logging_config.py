"""Central logging configuration for Vision Semantic Archive.

The UI (Streamlit) and the CLI (`python -m vsa ...`) both call
``configure_logging`` exactly once at startup to install a consistent format
and sensible per-package log levels (httpx/chromadb are chatty by default).
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any


_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> None:
    """Install dictConfig-based logging for the VSA process.

    Calling this more than once is a no-op to keep Streamlit's hot-reload
    from duplicating handlers on every script rerun.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = level if isinstance(level, int) else level.upper()
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            }
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": resolved_level,
            }
        },
        "root": {"level": resolved_level, "handlers": ["stderr"]},
        "loggers": {
            "httpx": {"level": "WARNING", "propagate": True},
            "httpcore": {"level": "WARNING", "propagate": True},
            "chromadb": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
        },
    }
    logging.config.dictConfig(config)
    _CONFIGURED = True
