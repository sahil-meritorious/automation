"""Merge per-chunk Whisper results into one transcript on the original timeline.

* Adds each chunk's offset to every timestamp.
* Uses chunk ownership regions (overlap midpoints) to drop duplicated words/segments.
* Filters typical Whisper hallucinations in silent passages.
* Guarantees every segment has word timings (synthesised if the API omitted them).
"""
from __future__ import annotations

import bisect
import difflib
import logging
import re
from typing import List, Optional, Tuple

from core.models import Segment, Transcript, Word
from modules.audio_chunker import AudioChunk

HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "please subscribe.", "subtitles by the amara.org community", "you", ".", "bye.", "bye!",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _is_hallucination(seg: dict) -> bool:
    text = _clean(seg.get("text", "")).lower()
    nsp = float(seg.get("no_speech_prob") or 0)
    logprob = float(seg.get("avg_logprob") or 0)
    comp = float(seg.get("compression_ratio") or 0)
    if not text:
        return True
    if comp > 2.8:  # heavy repetition loop
        return True
    if nsp > 0.85 and logprob < -0.6:
        return True
    if text in HALLUCINATIONS and nsp > 0.4:
        return True
    return False


def synthesize_words(seg: Segment) -> List[Word]:
    """Distribute a segment's duration over its words, weighted by length."""
    tokens = seg.text.split()
    if not tokens:
        return []
    dur = max(0.05, seg.end - seg.start)
    weights = [max(1, len(t)) for t in tokens]
    total = float(sum(weights))
    out, t = [], seg.start
    for tok, w in zip(tokens, weights):
        d = dur * w / total
        out.append(Word(word=tok, start=round(t, 3), end=round(t + d, 3)))
        t += d
    return out


def merge_chunk_results(
    results: List[Tuple[AudioChunk, dict]],
    video_id: str,
    duration: float,
    model: str,
    logger: Optional[logging.Logger] = None,
) -> Transcript:
    log = logger or logging.getLogger("shorts.merger")
    segments: List[Segment] = []
    words: List[Word] = []
    dropped: List[Tuple[float, float]] = []
    language: Optional[str] = None
    chunks_without_words = 0

    for chunk, raw in sorted(results, key=lambda r: r[0].index):
        off = chunk.start
        language = language or raw.get("language")
        own_s, own_e = chunk.own_start, chunk.own_end
        raw_segs = raw.get("segments") or []
        if not raw_segs and _clean(raw.get("text", "")):
            raw_segs = [{"start": 0.0, "end": chunk.duration, "text": raw["text"]}]

        for sg in raw_segs:
            st = off + float(sg.get("start", 0))
            en = off + float(sg.get("end", 0))
            if en <= st:
                en = st + 0.2
            mid = (st + en) / 2
            if not (own_s <= mid < own_e):
                continue
            if _is_hallucination(sg):
                dropped.append((st, en))
                continue
            segments.append(Segment(start=round(st, 3), end=round(min(en, duration), 3), text=_clean(sg.get("text", ""))))

        raw_words = raw.get("words") or []
        if not raw_words:
            chunks_without_words += 1
        for w in raw_words:
            text = _clean(w.get("word", ""))
            if not text:
                continue
            st = off + float(w.get("start", 0))
            en = max(st + 0.02, off + float(w.get("end", 0)))
            mid = (st + en) / 2
            if own_s <= mid < own_e:
                words.append(Word(word=text, start=round(st, 3), end=round(min(en, duration), 3),
                                  probability=w.get("probability")))

    if chunks_without_words:
        log.warning("%d chunk(s) returned no word timestamps; word timings will be estimated", chunks_without_words)

    segments.sort(key=lambda s: s.start)
    words.sort(key=lambda w: w.start)
    if dropped:
        words = [w for w in words if not any(a <= w.mid <= b for a, b in dropped)]
        log.info("Filtered %d likely-hallucinated segment(s)", len(dropped))

    # Remove near-duplicate segments around chunk boundaries
    deduped: List[Segment] = []
    for seg in segments:
        if deduped:
            prev = deduped[-1]
            if seg.start < prev.end - 0.3:
                ratio = difflib.SequenceMatcher(None, prev.text.lower(), seg.text.lower()).ratio()
                if ratio > 0.6:
                    if len(seg.text) > len(prev.text):
                        prev.text = seg.text
                    prev.end = max(prev.end, seg.end)
                    continue
        deduped.append(seg)
    segments = deduped

    # Assign every owned word to the segment containing it (or the nearest one)
    starts = [s.start for s in segments]
    orphans: List[Word] = []
    for w in words:
        i = bisect.bisect_right(starts, w.mid) - 1
        best = None
        if 0 <= i < len(segments) and segments[i].start - 0.05 <= w.mid <= segments[i].end + 0.05:
            best = segments[i]
        else:
            cands = [segments[j] for j in (i, i + 1) if 0 <= j < len(segments)]
            if cands:
                near = min(cands, key=lambda s: min(abs(s.start - w.mid), abs(s.end - w.mid)))
                if min(abs(near.start - w.mid), abs(near.end - w.mid)) <= 1.5:
                    best = near
        if best is not None:
            best.words.append(w)
        else:
            orphans.append(w)

    # Orphan words (speech whose segment was dropped at a boundary) become new segments
    if orphans:
        group: List[Word] = []
        for w in orphans + [None]:  # type: ignore[list-item]
            if w is not None and (not group or w.start - group[-1].end < 1.0):
                group.append(w)
                continue
            if group:
                segments.append(Segment(start=group[0].start, end=group[-1].end,
                                        text=" ".join(x.word for x in group), words=list(group)))
            group = [w] if w is not None else []
        segments.sort(key=lambda s: s.start)

    for seg in segments:
        seg.words.sort(key=lambda w: w.start)
        if not seg.words:
            seg.words = synthesize_words(seg)
        else:
            seg.start = min(seg.start, seg.words[0].start)
            seg.end = max(seg.end, seg.words[-1].end)

    segments = [s for s in segments if s.text]
    log.info("Merged transcript: %d segments, %d words, language=%s",
             len(segments), sum(len(s.words) for s in segments), language)
    return Transcript(video_id=video_id, duration=duration, language=language, model=model, transcript=segments)
