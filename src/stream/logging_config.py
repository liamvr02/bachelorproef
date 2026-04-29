"""
logging_config.py  -  /src/stream/logging_config.py
====================================================
Console + per-run timestamped file logging setup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger("stream")


def configure_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Configure console + per-run timestamped file logging.

    Creates a log file like:
        logs/stream_2026-04-13_14-32-10.log
    """

    numeric = getattr(logging, level.upper(), logging.INFO)

    class _TqdmHandler(logging.StreamHandler):
        """Write through tqdm.write() so log lines don't break progress bars."""
        def emit(self, record: logging.LogRecord) -> None:
            try:
                tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = log_path / f"stream_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-5s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Attach handlers to the *root* logger so that every named logger
    # (stream, lst_models.*, ingest.*, third-party) inherits the tqdm-safe
    # console handler.  Without this, anything outside the "stream" tree
    # logs straight to stderr and clobbers active tqdm bars.
    root = logging.getLogger()
    root.setLevel(numeric)

    # Drop the legacy "stream"-only handlers so log records aren't emitted
    # twice (once via the stream logger, once via root after propagation).
    stream_logger = logging.getLogger("stream")
    for h in list(stream_logger.handlers):
        stream_logger.removeHandler(h)
    stream_logger.propagate = True

    has_tqdm  = any(isinstance(h, _TqdmHandler)        for h in root.handlers)
    has_file  = any(isinstance(h, logging.FileHandler) for h in root.handlers)

    if not has_tqdm:
        console_handler = _TqdmHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if not has_file:
        file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party libraries that would otherwise flood DEBUG runs.
    for noisy in ("matplotlib", "PIL", "urllib3", "fiona", "pyogrio",
                  "rasterio", "shapely", "numexpr"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.INFO))

    root.info(f"Logging initialized -> {logfile}")