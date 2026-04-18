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

    root = logging.getLogger("stream")
    root.setLevel(numeric)

    # Avoid duplicate handlers if called multiple times
    if not root.handlers:
        # Console (tqdm-safe)
        console_handler = _TqdmHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        root.info(f"Logging initialized -> {logfile}")