"""Extract a mono 16 kHz WAV track (Whisper's native format) with FFmpeg."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.utils import FFmpegError, ensure_disk_space, media_duration, run_ffmpeg


def extract_audio(
    video_path: Path,
    out_path: Path,
    expected_duration: float,
    min_free_gb: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or logging.getLogger("shorts.audio")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        try:
            dur = media_duration(out_path)
            if expected_duration <= 0 or abs(dur - expected_duration) < max(2.0, expected_duration * 0.01):
                log.info("Audio already extracted (%s)", out_path.name)
                return out_path
            log.warning("Existing audio duration %.1fs != video %.1fs; re-extracting", dur, expected_duration)
        except FFmpegError:
            log.warning("Existing audio file is corrupt; re-extracting")

    # 16-bit mono 16 kHz PCM = 32 kB per second
    ensure_disk_space(out_path.parent, expected_duration * 32_000 * 1.1, min_free_gb, "audio extraction")
    tmp = out_path.with_name(out_path.stem + ".partial.wav")
    log.info("Extracting audio (mono, 16 kHz)…")
    run_ffmpeg(
        ["-i", str(video_path), "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)],
        timeout=max(600, expected_duration * 0.5),
        logger=log,
    )
    tmp.replace(out_path)
    log.info("Audio extracted: %.1f MB", out_path.stat().st_size / 1024**2)
    return out_path
