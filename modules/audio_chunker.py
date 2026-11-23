"""Split long audio into overlapping chunks that fit Groq's upload limit.

Each chunk records the part of the timeline it "owns" (boundary = middle of
the overlap), which the transcript merger uses to drop duplicated words.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from config import Settings
from core.utils import FFmpegError, atomic_write_json, fingerprint, read_json, run_ffmpeg


@dataclass
class AudioChunk:
    index: int
    start: float  # absolute start of the audio in this chunk (seconds)
    end: float
    own_start: float  # portion of the timeline this chunk is authoritative for
    own_end: float
    path: str = ""
    size_bytes: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_chunks(duration: float, chunk_seconds: float, overlap: float) -> List[AudioChunk]:
    """Chunk i covers [i*L, (i+1)*L + overlap]; ownership switches at the overlap midpoint."""
    if duration <= 0:
        return []
    chunk_seconds = max(30.0, float(chunk_seconds))
    overlap = max(0.0, min(float(overlap), chunk_seconds / 4))
    chunks: List[AudioChunk] = []
    i = 0
    while True:
        start = i * chunk_seconds
        if start >= duration:
            break
        end = min(duration, start + chunk_seconds + overlap)
        # Avoid a tiny last chunk: absorb a remainder shorter than 60 s into this chunk
        if duration - end < 60 and end < duration:
            end = duration
        chunks.append(AudioChunk(index=i, start=start, end=end, own_start=0, own_end=0))
        if end >= duration:
            break
        i += 1
    for idx, c in enumerate(chunks):
        c.own_start = 0.0 if idx == 0 else chunks[idx - 1].end - (chunks[idx - 1].end - c.start) / 2
        c.own_end = duration + 1.0 if idx == len(chunks) - 1 else c.end - (c.end - chunks[idx + 1].start) / 2
    return chunks


class AudioChunker:
    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self.s = settings
        self.log = logger or logging.getLogger("shorts.chunker")

    @property
    def max_bytes(self) -> int:
        return int(self.s.groq_max_upload_mb * 1024 * 1024 * self.s.upload_safety_margin)

    def _encode(self, wav: Path, chunk: AudioChunk, out: Path, fmt: str) -> None:
        codec = ["-c:a", "flac", "-compression_level", "8"] if fmt == "flac" else ["-c:a", "libmp3lame", "-b:a", "48k"]
        run_ffmpeg(
            ["-ss", f"{chunk.start:.3f}", "-t", f"{chunk.duration:.3f}", "-i", str(wav),
             "-ac", "1", "-ar", "16000", *codec, str(out)],
            timeout=600,
            logger=self.log,
        )

    def create_chunks(self, wav: Path, duration: float, out_dir: Path) -> List[AudioChunk]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.s.audio_chunk_format.lower() if self.s.audio_chunk_format.lower() in ("flac", "mp3") else "flac"
        plan = plan_chunks(duration, self.s.audio_chunk_seconds, self.s.audio_chunk_overlap)
        fp = fingerprint([duration, self.s.audio_chunk_seconds, self.s.audio_chunk_overlap, fmt, self.max_bytes])
        manifest_path = out_dir / "manifest.json"
        manifest = read_json(manifest_path, {})
        if manifest.get("fingerprint") == fp:
            chunks = [AudioChunk(**c) for c in manifest.get("chunks", [])]
            if chunks and all(Path(c.path).exists() for c in chunks):
                self.log.info("Reusing %d audio chunks", len(chunks))
                return chunks

        self.log.info("Splitting audio into %d chunk(s) of ~%ds (overlap %.0fs)",
                      len(plan), self.s.audio_chunk_seconds, self.s.audio_chunk_overlap)
        for chunk in plan:
            out = out_dir / f"chunk_{chunk.index:03d}.{fmt}"
            self._encode(wav, chunk, out, fmt)
            size = out.stat().st_size
            if size > self.max_bytes and fmt == "flac":
                self.log.warning("Chunk %d is %.1f MB (limit %.1f MB) — re-encoding as MP3",
                                 chunk.index, size / 1024**2, self.max_bytes / 1024**2)
                out.unlink(missing_ok=True)
                out = out_dir / f"chunk_{chunk.index:03d}.mp3"
                self._encode(wav, chunk, out, "mp3")
                size = out.stat().st_size
            if size > self.max_bytes:
                raise FFmpegError(
                    f"Audio chunk {chunk.index} is {size / 1024**2:.1f} MB, above the upload limit. "
                    f"Reduce AUDIO_CHUNK_SECONDS in .env."
                )
            if size < 1000:
                raise FFmpegError(f"Audio chunk {chunk.index} is empty — the audio track may be broken")
            chunk.path = str(out)
            chunk.size_bytes = size

        atomic_write_json(manifest_path, {"fingerprint": fp, "chunks": [asdict(c) for c in plan]})
        return plan
