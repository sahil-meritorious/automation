"""Groq Whisper transcription with rate limiting, retries and per-chunk caching.

Calls the Groq REST endpoint directly (OpenAI-compatible):
POST https://api.groq.com/openai/v1/audio/transcriptions
with response_format=verbose_json and word + segment timestamp granularities.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from config import Settings
from core.key_pool import NoValidKeyError, get_key_pool
from core.rate_limiter import RateLimiter, get_limiter
from core.utils import JobCancelled, atomic_write_json, interruptible_sleep, read_json, slugify
from modules.audio_chunker import AudioChunk

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
MIME = {".flac": "audio/flac", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg"}


class TranscriptionError(RuntimeError):
    pass


class ChunkTooLargeError(TranscriptionError):
    pass


def check_groq_key(api_key: str, timeout: float = 20) -> Tuple[bool, str]:
    """Cheap credential check (lists models)."""
    if not api_key:
        return False, "GROQ_API_KEY is not set"
    try:
        r = httpx.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    except httpx.HTTPError as exc:
        return False, f"Could not reach Groq: {exc}"
    if r.status_code == 200:
        ids = {m.get("id") for m in r.json().get("data", [])}
        has = [m for m in ("whisper-large-v3-turbo", "whisper-large-v3") if m in ids]
        return True, f"Groq key OK (whisper models available: {', '.join(has) or 'none listed'})"
    return False, f"Groq returned HTTP {r.status_code}: {r.text[:200]}"


class GroqTranscriber:
    def __init__(
        self,
        settings: Settings,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        if not settings.groq_keys:
            raise TranscriptionError("GROQ_API_KEY is not set. Add it to .env or enter it in the app sidebar.")
        self.s = settings
        self.log = logger or logging.getLogger("shorts.transcription")
        self.cancel_event = cancel_event
        self.pool = get_key_pool("groq whisper", settings.groq_keys)
        self.limiter: RateLimiter = self._limiter(self.pool.keys[0])

    def _limiter(self, key: str) -> RateLimiter:
        """Each key has its own Groq limits (when keys belong to different organisations)."""
        return get_limiter(
            f"groq-whisper {self.pool.keys.index(key)}",
            self.s.groq_requests_per_minute,
            [(3600, self.s.groq_audio_seconds_per_hour), (86400, self.s.groq_audio_seconds_per_day)],
        )

    # ------------------------------------------------------------------ helpers
    def _cache_path(self, cache_dir: Path, chunk: AudioChunk) -> Path:
        lang = self.s.transcription_language or "auto"
        return cache_dir / f"chunk_{chunk.index:03d}.{slugify(self.s.transcription_model)}.{lang}.json"

    @staticmethod
    def _retry_after(resp: httpx.Response, default: float) -> float:
        for header in ("retry-after", "x-ratelimit-reset-audio-seconds", "x-ratelimit-reset-requests"):
            val = resp.headers.get(header)
            if not val:
                continue
            try:
                return float(val)
            except ValueError:
                # Groq reset headers look like "1m26.4s" / "7.66s" / "2h3m"
                total, num = 0.0, ""
                for ch in val:
                    if ch.isdigit() or ch == ".":
                        num += ch
                    elif ch in "hms" and num:
                        total += float(num) * {"h": 3600, "m": 60, "s": 1}[ch]
                        num = ""
                if total:
                    return total
        return default

    def _request(self, chunk: AudioChunk) -> dict:
        path = Path(chunk.path)
        data: Dict[str, object] = {
            "model": self.s.transcription_model,
            "response_format": "verbose_json",
            "temperature": "0",
            "timestamp_granularities[]": ["word", "segment"],
        }
        if self.s.transcription_language:
            data["language"] = self.s.transcription_language
        if self.s.transcription_prompt:
            data["prompt"] = self.s.transcription_prompt[:800]
        timeout = httpx.Timeout(connect=30, read=600, write=600, pool=30)

        attempts = max(1, self.s.transcription_max_retries)
        backoff = 15.0
        for attempt in range(1, attempts + 1):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise JobCancelled("Job cancelled by user")
            # Groq bills a minimum of 10 seconds per request
            units = max(10.0, chunk.duration)
            try:
                key, _wait = self.pool.pick(lambda k: self._limiter(k).wait_time(units))
            except NoValidKeyError as exc:
                raise TranscriptionError(str(exc)) from exc
            self.limiter = self._limiter(key)
            self.limiter.acquire(units, self.cancel_event, self.log)
            headers = {"Authorization": f"Bearer {key}"}
            try:
                with path.open("rb") as fh:
                    resp = httpx.post(
                        GROQ_TRANSCRIBE_URL,
                        headers=headers,
                        data=data,
                        files={"file": (path.name, fh, MIME.get(path.suffix.lower(), "application/octet-stream"))},
                        timeout=timeout,
                    )
            except httpx.HTTPError as exc:
                self.log.warning("Chunk %d: network error %s (attempt %d/%d)", chunk.index, exc, attempt, attempts)
                interruptible_sleep(min(300, backoff), self.cancel_event)
                backoff *= 2
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    self.log.warning("Chunk %d: invalid JSON response; retrying", chunk.index)
                    continue
            if resp.status_code in (401, 403):
                if len(self.pool) > 1 and self.pool.invalid(key, f"HTTP {resp.status_code}"):
                    self.log.warning("Groq %s was rejected (HTTP %d) — skipping it", self.pool.label(key),
                                     resp.status_code)
                    continue
                raise TranscriptionError(f"Groq rejected the API key (HTTP {resp.status_code}). Check GROQ_API_KEY.")
            if resp.status_code == 413:
                raise ChunkTooLargeError(
                    f"Chunk {chunk.index} ({chunk.size_bytes / 1024**2:.1f} MB) exceeds Groq's upload limit. "
                    "Lower AUDIO_CHUNK_SECONDS or GROQ_MAX_UPLOAD_MB in .env."
                )
            if resp.status_code == 429:
                wait = min(3600.0, self._retry_after(resp, backoff) + 2)
                self.limiter.penalize(wait)
                if self.pool.cooldown(key, wait):
                    nxt, _w = self.pool.pick()
                    self.log.warning("Groq %s rate-limited on chunk %d (%.0fs) — switching to %s",
                                     self.pool.label(key), chunk.index, wait, self.pool.label(nxt))
                    continue  # does not count as a failed attempt worth backing off for
                self.log.warning("Groq rate limit hit on chunk %d (all %d key(s) limited) — pausing %.0fs (%s)",
                                 chunk.index, len(self.pool), wait, resp.text[:200])
                backoff = min(600, backoff * 2)
                continue
            if resp.status_code in (400, 404, 422):
                raise TranscriptionError(f"Groq rejected the request (HTTP {resp.status_code}): {resp.text[:400]}")
            # 5xx and anything else: back off and retry
            self.log.warning("Chunk %d: Groq HTTP %d (attempt %d/%d): %s",
                             chunk.index, resp.status_code, attempt, attempts, resp.text[:200])
            interruptible_sleep(min(300, backoff), self.cancel_event)
            backoff *= 2
        raise TranscriptionError(f"Transcription of chunk {chunk.index} failed after {attempts} attempts")

    # ------------------------------------------------------------------ public
    def transcribe_chunk(self, chunk: AudioChunk, cache_dir: Path) -> dict:
        cache = self._cache_path(cache_dir, chunk)
        cached = read_json(cache)
        if cached and cached.get("_chunk", {}).get("start") == chunk.start and cached.get("_chunk", {}).get("end") == chunk.end:
            return cached
        self.log.info("Transcribing chunk %d (%.0fs–%.0fs, %.1f MB) with %s",
                      chunk.index, chunk.start, chunk.end, chunk.size_bytes / 1024**2, self.s.transcription_model)
        result = self._request(chunk)
        result["_chunk"] = {"index": chunk.index, "start": chunk.start, "end": chunk.end}
        atomic_write_json(cache, result)
        n_words = len(result.get("words") or [])
        self.log.info("Chunk %d done: %d segments, %d words", chunk.index, len(result.get("segments") or []), n_words)
        return result

    def transcribe_all(
        self,
        chunks: List[AudioChunk],
        cache_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tuple[AudioChunk, dict]]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[int, Tuple[AudioChunk, dict]] = {}
        workers = max(1, min(self.s.transcription_workers, 4))
        done = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="groq") as pool:
            futures = {pool.submit(self.transcribe_chunk, c, cache_dir): c for c in chunks}
            try:
                for fut in as_completed(futures):
                    chunk = futures[fut]
                    results[chunk.index] = (chunk, fut.result())
                    done += 1
                    if progress:
                        progress(done / len(chunks), f"Transcribed {done}/{len(chunks)} audio chunks")
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
        return [results[i] for i in sorted(results)]
