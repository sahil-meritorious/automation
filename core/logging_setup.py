"""Logging configuration: console + rotating app log + one log file per job."""
from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False
_lock = threading.Lock()


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> None:
    global _configured
    with _lock:
        if _configured:
            return
        logs_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("shorts")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        root.addHandler(console)

        app_file = RotatingFileHandler(logs_dir / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        app_file.setLevel(logging.DEBUG)
        app_file.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(app_file)

        for noisy in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True


def get_job_logger(job_id: str, logs_dir: Path) -> logging.Logger:
    """A child of the ``shorts`` logger that also writes to logs/jobs/<job_id>.log."""
    logger = logging.getLogger(f"shorts.job.{job_id}")
    logger.setLevel(logging.DEBUG)
    path = logs_dir / "jobs" / f"{job_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


def close_job_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)


def job_log_path(job_id: str, logs_dir: Path) -> Path:
    return logs_dir / "jobs" / f"{job_id}.log"
