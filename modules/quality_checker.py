"""Quality control for rendered Shorts. A file that fails is never exported."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from core.utils import FFmpegError, run_ffmpeg
from modules.video_metadata import probe_video


@dataclass
class QCResult:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def to_dict(self) -> dict:
        return asdict(self)


def _speech_intervals(path: Path, duration: float) -> List[Tuple[float, float]]:
    """Non-silent regions of the audio track (via ffmpeg silencedetect)."""
    proc = run_ffmpeg(["-i", str(path), "-vn", "-af", "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"],
                      timeout=300)
    silences: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    if start is not None:
        silences.append((start, duration))
    speech, cursor = [], 0.0
    for a, b in silences:
        if a > cursor:
            speech.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        speech.append((cursor, duration))
    return speech


def _overlap(spans: Sequence[Tuple[float, float]], regions: Sequence[Tuple[float, float]]) -> float:
    total = sum(max(0.0, b - a) for a, b in spans)
    if total <= 0:
        return 1.0
    covered = 0.0
    for a, b in spans:
        for c, d in regions:
            covered += max(0.0, min(b, d) - max(a, c))
    return covered / total


class QualityChecker:
    def __init__(
        self,
        width: int,
        height: int,
        min_duration: float,
        max_duration: float,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.width, self.height = width, height
        self.min_duration, self.max_duration = min_duration, max_duration
        self.log = logger or logging.getLogger("shorts.qc")

    def check(
        self,
        path: Path,
        expected_duration: float,
        caption_spans: Optional[Sequence[Tuple[float, float]]] = None,
        captions_expected: bool = False,
    ) -> QCResult:
        r = QCResult()
        if not path.exists():
            r.fail("output file does not exist")
            return r
        size = path.stat().st_size
        r.info["size_mb"] = round(size / 1024**2, 2)
        if size < 20_000:
            r.fail(f"output file is too small ({size} bytes)")
            return r

        try:
            meta = probe_video(path)
        except FFmpegError as exc:
            r.fail(f"ffprobe cannot read the file: {exc}")
            return r
        r.info.update({k: meta[k] for k in ("duration", "width", "height", "fps", "vcodec", "acodec")})

        if not meta["has_video"]:
            r.fail("no video stream")
        if not meta["has_audio"]:
            r.fail("no audio stream")
        if meta["vcodec"] != "h264":
            r.fail(f"video codec is {meta['vcodec']}, expected h264")
        if meta["has_audio"] and meta["acodec"] != "aac":
            r.fail(f"audio codec is {meta['acodec']}, expected aac")
        if (meta["width"], meta["height"]) != (self.width, self.height):
            r.fail(f"resolution {meta['width']}x{meta['height']} != {self.width}x{self.height}")
        if meta["height"] and abs(meta["width"] / meta["height"] - 9 / 16) > 0.01:
            r.fail("aspect ratio is not 9:16")

        dur = meta["duration"]
        if dur < max(1.0, self.min_duration - 1.0):
            r.fail(f"duration {dur:.1f}s is below the minimum {self.min_duration:.0f}s")
        if dur > self.max_duration + 1.5:
            r.fail(f"duration {dur:.1f}s exceeds the maximum {self.max_duration:.0f}s")
        if abs(dur - expected_duration) > 1.0:
            r.fail(f"duration {dur:.1f}s differs from the planned {expected_duration:.1f}s")
        if meta["video_duration"] and meta["audio_duration"]:
            drift = abs(meta["video_duration"] - meta["audio_duration"])
            r.info["av_duration_drift"] = round(drift, 3)
            if drift > 0.5:
                r.fail(f"audio/video length mismatch of {drift:.2f}s (possible desync)")

        # Full decode test catches corrupt frames
        try:
            proc = run_ffmpeg(["-v", "error", "-i", str(path), "-f", "null", "-"], timeout=600)
            errs = [l for l in (proc.stderr or "").splitlines() if l.strip()]
            if errs:
                r.warnings.append(f"decoder reported {len(errs)} issue(s): {errs[0][:120]}")
        except FFmpegError as exc:
            r.fail(f"file does not decode cleanly: {str(exc)[:200]}")

        # Subtitle synchronisation
        if captions_expected:
            spans = list(caption_spans or [])
            if not spans:
                r.warnings.append("no caption events were generated (no speech in clip?)")
            else:
                out_of_range = [s for s in spans if s[0] < -0.05 or s[1] > dur + 0.1 or s[1] < s[0]]
                if out_of_range:
                    r.fail(f"{len(out_of_range)} caption event(s) fall outside the clip timeline")
                try:
                    speech = _speech_intervals(path, dur)
                    cov = _overlap(spans, speech)
                    r.info["caption_speech_overlap"] = round(cov, 3)
                    if cov < 0.5:
                        r.warnings.append(f"only {cov:.0%} of caption time overlaps detected speech — check sync")
                except FFmpegError as exc:
                    r.warnings.append(f"subtitle sync check skipped: {exc}")
        return r
