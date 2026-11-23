"""LLM-based highlight detection over long transcripts.

The transcript is split into overlapping time windows. Each window is analysed
independently (results cached on disk), producing candidate clips on the
original timeline. Ranking/dedup across windows happens in highlight_ranker.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from pydantic import ValidationError

from config import Settings
from modules.campaign_brief import brief_prompt
from core.llm_client import LLMClient, LLMError, RequestTooLargeError, fill_template, load_prompt
from core.models import ClipCandidate, Transcript, VideoInfo
from core.utils import JobCancelled, atomic_write_json, fingerprint, read_json


def build_windows(duration: float, size: float, overlap: float) -> List[Tuple[float, float]]:
    size = max(120.0, float(size))
    overlap = max(0.0, min(float(overlap), size / 3))
    step = size - overlap
    windows, start = [], 0.0
    while start < duration:
        end = min(duration, start + size)
        # merge a short tail into the previous window
        if windows and end - start < size * 0.35 and end >= duration:
            windows[-1] = (windows[-1][0], end)
            break
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def format_transcript_window(transcript: Transcript, start: float, end: float) -> str:
    lines = []
    for seg in transcript.segments_between(start, end):
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
    return "\n".join(lines)


class HighlightDetector:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.log = logger or logging.getLogger("shorts.detector")
        self.cancel_event = cancel_event
        self.system, self.user_template = load_prompt(settings.prompts_dir, "highlight_detection.txt")

    def _validator(self, w_start: float, w_end: float, duration: float) -> Callable[[Any], List[ClipCandidate]]:
        def validate(data: Any) -> List[ClipCandidate]:
            if isinstance(data, list):
                data = {"clips": data}
            if not isinstance(data, dict) or "clips" not in data or not isinstance(data["clips"], list):
                raise ValueError('expected an object like {"clips": [...]}')
            clips, errors = [], []
            for i, raw in enumerate(data["clips"]):
                try:
                    c = ClipCandidate.model_validate(raw)
                except ValidationError as exc:
                    errors.append(f"clip {i}: {exc.errors()[0].get('msg')}")
                    continue
                # discard hallucinated timestamps far outside this window
                if c.clip_start < w_start - 45 or c.clip_end > w_end + 45 or c.clip_start >= duration:
                    errors.append(f"clip {i}: timestamps {c.clip_start}-{c.clip_end} outside section")
                    continue
                c.clip_start = max(0.0, c.clip_start)
                c.clip_end = min(duration, c.clip_end)
                clips.append(c)
            if data["clips"] and not clips:
                raise ValueError("no valid clips: " + "; ".join(errors[:5]))
            if errors:
                self.log.debug("Discarded invalid clips: %s", errors)
            return clips

        return validate

    def _analyse_window(
        self, idx: int, total: int, w: Tuple[float, float], transcript: Transcript, info: VideoInfo, cache_dir: Path
    ) -> List[ClipCandidate]:
        return self._analyse_range(idx, total, w[0], w[1], transcript, info, cache_dir)

    MIN_SPLIT_SECONDS = 150.0

    def _split(self, idx: int, total: int, start: float, end: float, transcript: Transcript, info: VideoInfo,
               cache_dir: Path, part: str, why: str) -> List[ClipCandidate]:
        """Analyse a section in two overlapping halves (prompt too large for the provider's token limit)."""
        mid = (start + end) / 2
        ov = min(30.0, (end - start) / 10)
        self.log.info("Section %d%s (%.0f–%.0fs) %s — splitting into two halves", idx + 1, part, start, end, why)
        clips = self._analyse_range(idx, total, start, mid + ov, transcript, info, cache_dir, part + "a")
        clips += self._analyse_range(idx, total, mid - ov, end, transcript, info, cache_dir, part + "b")
        return clips

    def _analyse_range(
        self, idx: int, total: int, start: float, end: float, transcript: Transcript, info: VideoInfo,
        cache_dir: Path, part: str = "",
    ) -> List[ClipCandidate]:
        text = format_transcript_window(transcript, start, end)
        if len(text.split()) < 25:
            self.log.info("Section %d/%d (%.0f–%.0fs): too little speech, skipped", idx + 1, total, start, end)
            return []
        s = self.s
        user = fill_template(
            self.user_template,
            TITLE=info.title or info.video_id,
            CONTEXT=s.content_context or "general video", AUDIENCE=s.target_audience, BRIEF=brief_prompt(s),
            DURATION=f"{transcript.duration:.0f}",
            INDEX=idx + 1,
            TOTAL=total,
            START=f"{start:.0f}",
            END=f"{end:.0f}",
            MIN_DUR=f"{s.min_clip_duration:.0f}",
            MAX_DUR=f"{s.max_clip_duration:.0f}",
            PREF_MIN=f"{s.preferred_min_duration:.0f}",
            PREF_MAX=f"{s.preferred_max_duration:.0f}",
            HARD_MAX=f"{s.hard_max_duration:.0f}",
            MAX_CANDIDATES=s.candidates_per_window,
            TRANSCRIPT=text,
        )
        fp = fingerprint([self.llm.label, self.system, user, s.llm_temperature])
        cache = cache_dir / f"window_{idx:03d}{part}.json"
        cached = read_json(cache)
        if cached and cached.get("fingerprint") == fp:
            clips = [ClipCandidate.model_validate(c) for c in cached.get("clips", [])]
            self.log.info("Section %d/%d: %d cached candidate(s)", idx + 1, total, len(clips))
            return clips

        can_split = end - start >= self.MIN_SPLIT_SECONDS
        if can_split and not self.llm.fits(self.system, user):
            return self._split(idx, total, start, end, transcript, info, cache_dir, part,
                               "is larger than the per-request token budget")
        self.log.info("Analysing section %d/%d%s (%s–%s) with %s",
                      idx + 1, total, part, f"{start:.0f}s", f"{end:.0f}s", self.llm.label)
        try:
            clips = self.llm.chat_json(
                self.system,
                user,
                validate=self._validator(start, end, transcript.duration),
                attempts=3,
                what=f"Highlight detection section {idx + 1}{part}",
                cancel_event=self.cancel_event,
            )
        except RequestTooLargeError:
            if not can_split:
                raise
            return self._split(idx, total, start, end, transcript, info, cache_dir, part,
                               "was rejected as too large by the provider")
        clips = sorted(clips, key=lambda c: c.score, reverse=True)[: s.candidates_per_window]
        for j, c in enumerate(clips):
            c.window_index = idx
            c.id = f"w{idx:03d}{part}c{j}"
        atomic_write_json(cache, {
            "fingerprint": fp, "window": [start, end], "model": self.llm.label,
            "clips": [c.model_dump() for c in clips],
        })
        self.log.info("Section %d/%d%s: %d candidate(s)%s", idx + 1, total, part, len(clips),
                      (" — best: %.0f '%s'" % (clips[0].score, clips[0].title)) if clips else "")
        return clips

    def detect(
        self,
        transcript: Transcript,
        info: VideoInfo,
        cache_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[ClipCandidate]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        windows = build_windows(transcript.duration, self.s.analysis_window_seconds, self.s.analysis_overlap_seconds)
        self.log.info("Highlight detection: %d section(s) of ~%ds", len(windows), self.s.analysis_window_seconds)
        all_clips: List[ClipCandidate] = []
        failures: List[str] = []
        done = 0
        workers = max(1, min(self.s.llm_max_concurrency, 4))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm") as pool:
            futures = {
                pool.submit(self._analyse_window, i, len(windows), w, transcript, info, cache_dir): i
                for i, w in enumerate(windows)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    all_clips.extend(fut.result())
                except JobCancelled:
                    for f in futures:
                        f.cancel()
                    raise
                except (LLMError, Exception) as exc:  # one bad section must not kill the job
                    failures.append(f"section {i + 1}: {exc}")
                    self.log.error("Section %d failed: %s", i + 1, exc)
                done += 1
                if progress:
                    progress(done / len(windows), f"Analysed {done}/{len(windows)} transcript sections")

        if failures and len(failures) == len(windows):
            raise LLMError("Highlight detection failed for every section: " + failures[0])
        if failures:
            self.log.warning("%d section(s) failed and were skipped; re-run the job to retry them", len(failures))
        atomic_write_json(cache_dir / "candidates_raw.json", [c.model_dump() for c in all_clips])
        return all_clips
