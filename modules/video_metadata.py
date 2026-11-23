"""Read technical metadata (duration, resolution, fps, streams) with ffprobe."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.utils import FFmpegError, ffprobe_json


def _parse_fps(rate: str) -> float:
    try:
        if "/" in rate:
            num, den = rate.split("/")
            return float(num) / float(den) if float(den) else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path | str) -> Dict[str, Any]:
    """Return duration, width, height, fps, codecs and stream presence for a media file."""
    path = Path(path)
    if not path.exists():
        raise FFmpegError(f"File not found: {path}")
    info = ffprobe_json(path)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})

    duration = float(fmt.get("duration") or 0.0)
    if not duration:
        for s in (video, audio):
            if s and s.get("duration"):
                duration = float(s["duration"])
                break

    width = int(video.get("width", 0)) if video else 0
    height = int(video.get("height", 0)) if video else 0
    # Respect rotation metadata (phone footage)
    rotation = 0
    if video:
        for side in video.get("side_data_list", []) or []:
            if "rotation" in side:
                rotation = abs(int(side["rotation"]))
        rotation = rotation or abs(int(video.get("tags", {}).get("rotate", 0) or 0))
    if rotation in (90, 270):
        width, height = height, width

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0") if video else 0.0,
        "vcodec": video.get("codec_name", "") if video else "",
        "acodec": audio.get("codec_name", "") if audio else "",
        "has_video": video is not None,
        "has_audio": audio is not None,
        "video_duration": float(video.get("duration") or 0) if video else 0.0,
        "audio_duration": float(audio.get("duration") or 0) if audio else 0.0,
        "size_bytes": int(fmt.get("size") or path.stat().st_size),
        "format_name": fmt.get("format_name", ""),
    }
