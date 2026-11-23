"""Frame-accurate extraction of a source segment into a high-quality intermediate clip."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.utils import FFmpegError, ensure_disk_space, run_ffmpeg
from modules.video_metadata import probe_video


def extract_clip(
    source: Path,
    start: float,
    end: float,
    out_path: Path,
    min_free_gb: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or logging.getLogger("shorts.extractor")
    duration = end - start
    if duration <= 0:
        raise ValueError("clip end must be after start")

    if out_path.exists():
        try:
            meta = probe_video(out_path)
            if meta["has_video"] and abs(meta["duration"] - duration) < 0.5:
                return out_path
        except FFmpegError:
            pass

    ensure_disk_space(out_path.parent, duration * 2_000_000, min_free_gb, "clip extraction")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial.mp4")
    # Input seeking + re-encode = fast and frame accurate. CRF 16 keeps quality for the second encode.
    run_ffmpeg(
        [
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
            str(tmp),
        ],
        timeout=max(300, duration * 10),
        logger=log,
    )
    tmp.replace(out_path)
    return out_path
