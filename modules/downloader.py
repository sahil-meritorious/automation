"""Video acquisition: YouTube via yt-dlp (for content you are authorised to use) or local files.

* Validates the URL and rejects playlists / live streams.
* Reads metadata first (title, duration, resolution) without downloading.
* Caches downloads in downloads/<video_id>/ so a video is never fetched twice.
* Resumes partial downloads and retries on failure.
This module does not bypass access controls, logins or DRM.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple

from config import Settings
from core.models import VideoInfo
from core.utils import (
    DiskSpaceError,
    FFmpegError,
    JobCancelled,
    atomic_write_json,
    ensure_disk_space,
    file_fingerprint,
    read_json,
    retry_call,
    slugify,
)
from modules.video_metadata import probe_video

YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts"}

ProgressFn = Callable[[float, str], None]


class DownloadError(RuntimeError):
    pass


def parse_youtube_url(url: str) -> Optional[str]:
    """Return the 11-character video id, or None if this is not a YouTube video URL."""
    m = YOUTUBE_ID_RE.search((url or "").strip())
    return m.group(1) if m else None


def is_local_source(source: str) -> bool:
    p = Path(source.strip().strip('"'))
    return p.suffix.lower() in VIDEO_EXTENSIONS and p.exists()


def validate_source(source: str) -> Tuple[bool, str]:
    """(ok, message) for UI/CLI validation."""
    source = (source or "").strip()
    if not source:
        return False, "Please enter a YouTube URL or a local video file path."
    if is_local_source(source):
        return True, "Local video file"
    if source.lower().startswith(("http://", "https://")) or "youtu" in source.lower():
        if parse_youtube_url(source):
            return True, "YouTube video URL"
        if "list=" in source:
            return False, "Playlist URLs are not supported — paste a single video URL."
        return False, "Not a recognised YouTube video URL (expected youtube.com/watch?v=… or youtu.be/…)."
    return False, "Input is neither a YouTube URL nor an existing video file."


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_deprecation_reported = False


class _YtdlpLogger:
    """Routes yt-dlp output into our logs: strips colour codes and downgrades
    deprecation notices (e.g. "Python 3.10 is deprecated") to one INFO line."""

    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger

    @staticmethod
    def _clean(msg: str) -> str:
        return _ANSI_RE.sub("", str(msg)).strip()

    def _deprecation(self, msg: str) -> bool:
        global _deprecation_reported
        if "Deprecated Feature" not in msg:
            return False
        if not _deprecation_reported:
            _deprecation_reported = True
            self.log.info("yt-dlp notice (not an error): %s", msg.split(":", 1)[-1].strip())
        return True

    def debug(self, msg: str) -> None:
        msg = self._clean(msg)
        if not msg.startswith("[download]"):
            self.log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        self.log.debug("yt-dlp: %s", self._clean(msg))

    def warning(self, msg: str) -> None:
        msg = self._clean(msg)
        if not self._deprecation(msg):
            self.log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        msg = self._clean(msg)
        if not self._deprecation(msg):
            self.log.error("yt-dlp: %s", msg)


class Downloader:
    def __init__(
        self,
        settings: Settings,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        self.log = logger or logging.getLogger("shorts.downloader")
        self.cancel_event = cancel_event

    # ------------------------------------------------------------------ public
    def resolve(self, source: str, progress: Optional[ProgressFn] = None) -> VideoInfo:
        """Return VideoInfo with a local ``source_path`` (downloading if necessary)."""
        source = source.strip().strip('"')
        ok, msg = validate_source(source)
        if not ok:
            raise DownloadError(msg)
        if is_local_source(source):
            return self._from_local(Path(source))
        return self._from_youtube(source, progress)

    # ------------------------------------------------------------------ local
    def _from_local(self, path: Path) -> VideoInfo:
        path = path.resolve()
        meta = probe_video(path)
        if not meta["has_video"]:
            raise DownloadError(f"{path.name} has no video stream")
        video_id = f"local_{slugify(path.stem, 30)}_{file_fingerprint(path)[:8]}"
        self.log.info("Using local file %s (%.0fs, %dx%d)", path.name, meta["duration"], meta["width"], meta["height"])
        return VideoInfo(
            video_id=video_id,
            source=str(path),
            title=path.stem,
            duration=meta["duration"],
            width=meta["width"],
            height=meta["height"],
            fps=meta["fps"],
            has_audio=meta["has_audio"],
            source_path=str(path),
        )

    # ------------------------------------------------------------------ youtube
    def _ydl(self, extra: Optional[dict] = None):
        try:
            import yt_dlp
        except ImportError as exc:
            raise DownloadError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc
        opts = {
            "quiet": True,
            "no_warnings": False,
            "noprogress": True,
            "noplaylist": True,
            "logger": _YtdlpLogger(self.log),
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 30,
            "ffmpeg_location": None if self.s.ffmpeg_path == "ffmpeg" else self.s.ffmpeg_path,
        }
        opts = {k: v for k, v in opts.items() if v is not None}
        if extra:
            opts.update(extra)
        return yt_dlp.YoutubeDL(opts)

    def fetch_info(self, url: str) -> dict:
        def _get() -> dict:
            with self._ydl() as ydl:
                info = ydl.extract_info(url, download=False)
                return ydl.sanitize_info(info)

        info = retry_call(_get, attempts=self.s.download_retries, base_delay=5, logger=self.log,
                          what="Fetching video metadata", cancel_event=self.cancel_event)
        if not info:
            raise DownloadError("yt-dlp returned no metadata")
        if info.get("_type") == "playlist":
            raise DownloadError("Playlists are not supported — paste a single video URL.")
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise DownloadError("Live or upcoming streams cannot be processed. Wait until the VOD is available.")
        return info

    def _find_cached(self, folder: Path, video_id: str) -> Optional[Path]:
        if not folder.exists():
            return None
        for p in sorted(folder.glob(f"{video_id}.*")):
            if p.suffix.lower() in VIDEO_EXTENSIONS and ".part" not in p.name and ".f" not in p.stem[len(video_id):]:
                try:
                    meta = probe_video(p)
                    if meta["has_video"] and meta["duration"] > 0:
                        return p
                except FFmpegError:
                    self.log.warning("Cached file %s is unreadable; re-downloading", p.name)
        return None

    def _from_youtube(self, url: str, progress: Optional[ProgressFn]) -> VideoInfo:
        video_id = parse_youtube_url(url)
        assert video_id
        folder = self.s.downloads_dir / video_id
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / "info.json"

        info = read_json(info_path)
        cached = self._find_cached(folder, video_id)
        if cached and info:
            self.log.info("Source video already downloaded: %s", cached.name)
        else:
            self.log.info("Fetching metadata for %s", url)
            info = self.fetch_info(url)
            atomic_write_json(info_path, {k: info.get(k) for k in (
                "id", "title", "duration", "width", "height", "fps", "uploader", "channel", "upload_date",
                "webpage_url", "description", "tags", "categories", "filesize_approx")})
            self.log.info("Title: %s | duration: %ss", info.get("title"), info.get("duration"))
            if not cached:
                cached = self._download(url, folder, video_id, info, progress)

        meta = probe_video(cached)
        return VideoInfo(
            video_id=video_id,
            source=url,
            title=str(info.get("title") or video_id),
            duration=meta["duration"] or float(info.get("duration") or 0),
            width=meta["width"],
            height=meta["height"],
            fps=meta["fps"],
            has_audio=meta["has_audio"],
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            upload_date=str(info.get("upload_date") or ""),
            webpage_url=str(info.get("webpage_url") or url),
            source_path=str(cached),
        )

    def _download(self, url: str, folder: Path, video_id: str, info: dict, progress: Optional[ProgressFn]) -> Path:
        duration = float(info.get("duration") or 0)
        estimate = float(info.get("filesize_approx") or 0) or duration * 0.8 * 1024 * 1024  # ~6.5 Mbit/s
        ensure_disk_space(folder, estimate * 1.3, self.s.min_free_disk_gb, "video download")

        h = self.s.max_download_height
        fmt = (
            f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={h}]+ba/"
            f"b[height<={h}]/b"
        )

        def hook(d: dict) -> None:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise JobCancelled("Job cancelled by user")
            if progress and d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done = d.get("downloaded_bytes") or 0
                if total:
                    speed = (d.get("speed") or 0) / 1024 / 1024
                    progress(min(0.99, done / total), f"Downloading {done / 1024**2:.0f}/{total / 1024**2:.0f} MB ({speed:.1f} MB/s)")

        def _do() -> None:
            with self._ydl({
                "format": fmt,
                "merge_output_format": "mp4",
                "outtmpl": str(folder / f"{video_id}.%(ext)s"),
                "continuedl": True,
                "concurrent_fragment_downloads": 4,
                "progress_hooks": [hook],
            }) as ydl:
                ret = ydl.download([url])
                if ret != 0:
                    raise DownloadError(f"yt-dlp exited with code {ret}")

        self.log.info("Downloading video (max %sp)…", h)
        try:
            retry_call(_do, attempts=self.s.download_retries, base_delay=10, logger=self.log, what="Video download",
                       exceptions=(Exception,), cancel_event=self.cancel_event)
        except (JobCancelled, DiskSpaceError):
            raise
        except Exception as exc:
            raise DownloadError(
                f"Download failed: {exc}. If YouTube blocks the request, update yt-dlp "
                f"(pip install -U \"yt-dlp[default]\") and make sure a JavaScript runtime such as Deno is installed, "
                f"or download the video yourself and pass the local file path instead."
            ) from exc

        path = self._find_cached(folder, video_id)
        if not path:
            raise DownloadError("Download finished but no playable video file was found")
        self.log.info("Downloaded %s (%.1f MB)", path.name, path.stat().st_size / 1024**2)
        return path
