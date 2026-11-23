"""Shared helpers: subprocess/FFmpeg wrappers, retries, disk checks, JSON IO."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

log = logging.getLogger("shorts.utils")
T = TypeVar("T")


class FFmpegError(RuntimeError):
    """An external command (ffmpeg/ffprobe) failed."""


class DiskSpaceError(RuntimeError):
    """Not enough free disk space for the next step."""


class JobCancelled(Exception):
    """Raised when the user cancels a running job."""


_BINARIES = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}


def configure_binaries(ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
    _BINARIES["ffmpeg"] = ffmpeg_path or "ffmpeg"
    _BINARIES["ffprobe"] = ffprobe_path or "ffprobe"


def ffmpeg_bin() -> str:
    return _BINARIES["ffmpeg"]


def ffprobe_bin() -> str:
    return _BINARIES["ffprobe"]


# --------------------------------------------------------------------- subprocess
def run_cmd(
    args: Sequence[Any],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> subprocess.CompletedProcess:
    logger = logger or log
    cmd = [str(a) for a in args]
    logger.debug("Running: %s", " ".join(cmd))
    creationflags = 0
    if os.name == "nt":  # do not pop up console windows on Windows
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            f"Executable not found: '{cmd[0]}'. Install it and make sure it is on PATH "
            f"(or set FFMPEG_PATH / FFPROBE_PATH in .env)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"Command timed out after {timeout}s: {Path(cmd[0]).name}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-30:])
        raise FFmpegError(f"{Path(cmd[0]).name} failed (exit code {proc.returncode}):\n{tail}")
    return proc


def run_ffmpeg(args: Sequence[Any], **kwargs) -> subprocess.CompletedProcess:
    return run_cmd([ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args], **kwargs)


def ffprobe_json(path: Path | str) -> dict:
    proc = run_cmd(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path}") from exc


def media_duration(path: Path | str) -> float:
    info = ffprobe_json(path)
    dur = info.get("format", {}).get("duration")
    if dur is None:
        for s in info.get("streams", []):
            if s.get("duration"):
                dur = s["duration"]
                break
    return float(dur or 0.0)


def check_binaries() -> dict:
    versions = {}
    for name in ("ffmpeg", "ffprobe"):
        proc = run_cmd([_BINARIES[name], "-version"], timeout=30)
        versions[name] = (proc.stdout or "").splitlines()[0] if proc.stdout else "unknown"
    return versions


def escape_filter_path(path: Path | str) -> str:
    """Escape a path for use inside an FFmpeg filtergraph option value."""
    p = str(Path(path).resolve()).replace("\\", "/")
    p = p.replace(":", r"\:").replace("'", r"'\''")
    return f"'{p}'"


# --------------------------------------------------------------------- disk
def ensure_disk_space(path: Path, required_bytes: float, min_free_gb: float, what: str = "") -> None:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    needed = required_bytes + min_free_gb * 1024**3
    if usage.free < needed:
        raise DiskSpaceError(
            f"Not enough disk space{(' for ' + what) if what else ''}: "
            f"{usage.free / 1024**3:.2f} GB free, ~{needed / 1024**3:.2f} GB needed "
            f"(includes MIN_FREE_DISK_GB={min_free_gb})."
        )


def remove_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        log.warning("Could not remove %s: %s", path, exc)


# --------------------------------------------------------------------- text / time
def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    text = text[:max_len].rstrip("-")
    return text or "short"


def strip_unsupported_chars(text: str) -> str:
    """Drop emoji / astral-plane characters that typical caption fonts cannot render."""
    return "".join(ch for ch in (text or "") if ord(ch) < 0x10000 and unicodedata.category(ch) != "So").strip()


def fmt_ts(seconds: float, millis: bool = True) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}" if millis else f"{h:02d}:{m:02d}:{int(s):02d}"


def parse_timestamp(value: Any) -> float:
    """Accept seconds (int/float/str) or 'mm:ss(.f)' / 'hh:mm:ss(.f)'."""
    if value is None:
        raise ValueError("timestamp is missing")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().rstrip("s")
    if not text:
        raise ValueError("timestamp is empty")
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        total = 0.0
        for p in parts:
            total = total * 60 + p
        return total
    return float(text)


# --------------------------------------------------------------------- hashing / json
def fingerprint(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    return fingerprint([str(path.resolve()), st.st_size, int(st.st_mtime)])


_write_lock = threading.Lock()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    with _write_lock:
        os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


# --------------------------------------------------------------------- retry
def retry_call(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    logger: Optional[logging.Logger] = None,
    what: str = "operation",
    cancel_event: Optional[threading.Event] = None,
) -> T:
    logger = logger or log
    last: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203
            last = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (0.8 + random.random() * 0.4)
            logger.warning("%s failed (attempt %d/%d): %s — retrying in %.1fs", what, attempt, attempts, exc, delay)
            interruptible_sleep(delay, cancel_event)
    assert last is not None
    raise last


def interruptible_sleep(seconds: float, cancel_event: Optional[threading.Event] = None) -> None:
    if cancel_event is None:
        time.sleep(seconds)
        return
    if cancel_event.wait(seconds):
        raise JobCancelled("Job cancelled by user")
