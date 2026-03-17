"""Centralized logging setup with request/run context."""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
_configured = False


class _ContextFilter(logging.Filter):
    """Inject request/run ids into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get("-")
        record.run_id = _run_id_var.get("-")
        return True


@contextlib.contextmanager
def logging_context(request_id: Optional[str] = None, run_id: Optional[str] = None) -> Iterator[None]:
    """Temporarily set request/run ids for logs emitted inside the context."""
    req_token = _request_id_var.set(request_id) if request_id is not None else None
    run_token = _run_id_var.set(run_id) if run_id is not None else None
    try:
        yield
    finally:
        if run_token is not None:
            _run_id_var.reset(run_token)
        if req_token is not None:
            _request_id_var.reset(req_token)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging handlers once per process."""
    global _configured
    if _configured:
        return

    log_level = getattr(logging, str(level).upper(), logging.INFO)
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | req=%(request_id)s run=%(run_id)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    context_filter = _ContextFilter()

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    console.addFilter(context_filter)
    root.addHandler(console)

    out_file = RotatingFileHandler(
        log_dir / "backend.out.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    out_file.setLevel(log_level)
    out_file.setFormatter(formatter)
    out_file.addFilter(context_filter)
    root.addHandler(out_file)

    err_file = RotatingFileHandler(
        log_dir / "backend.err.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    err_file.setLevel(logging.ERROR)
    err_file.setFormatter(formatter)
    err_file.addFilter(context_filter)
    root.addHandler(err_file)

    # Keep framework logs, reduce noisy HTTP client internals by default.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    _configured = True

