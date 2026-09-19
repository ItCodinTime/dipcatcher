"""Structured logging. Never log secrets."""

from __future__ import annotations

import logging
import os
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(**binds: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger().bind(**binds)  # type: ignore[no-any-return]
