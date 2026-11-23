"""Global ranking and selection of highlight candidates.

1. Refine boundaries: snap to word/sentence boundaries, enforce duration limits
   while preserving setup → hook → payoff.
2. Combine the LLM score with weighted sub-scores.
3. Remove duplicates (overlapping candidates from overlapping windows).
4. Optional global LLM re-rank comparing all strong candidates side by side.
5. Keep only candidates above MIN_SCORE, up to NUM_SHORTS. No filler.
"""
from __future__ import annotations

import bisect
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Settings
from modules.campaign_brief import brief_prompt
from core.llm_client import LLMClient, LLMError, fill_template, load_prompt
from core.models import ClipCandidate, Transcript, VideoInfo
from core.utils import JobCancelled, atomic_write_json, fingerprint, read_json

SUBSCORE_WEIGHTS = {
    "hook": 0.25,
    "entertainment": 0.20,
    "retention": 0.15,
    "standalone": 0.15,
    "emotion": 0.10,
    "originality": 0.10,
    "clarity": 0.05,
}
SENTENCE_END = (".", "!", "?", "…")


def overlap_ratio(a: ClipCandidate, b: ClipCandidate) -> float:
    inter = max(0.0, min(a.clip_end, b.clip_end) - max(a.clip_start, b.clip_start))
    shorter = max(0.1, min(a.duration, b.duration))
    return inter / shorter


def is_duplicate(a: ClipCandidate, b: ClipCandidate) -> bool:
    return overlap_ratio(a, b) > 0.4 or abs(a.hook - b.hook) < 4.0


class HighlightRanker:
    def __init__(
        self,
        settings: Settings,
        llm: Optional[LLMClient] = None,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.log = logger or logging.getLogger("shorts.ranker")
        self.cancel_event = cancel_event

    # ------------------------------------------------------------ boundaries
    def refine(self, c: ClipCandidate, transcript: Transcript) -> Optional[ClipCandidate]:
        s = self.s
        words = transcript.all_words()
        starts = [w.start for w in words]
        duration = transcript.duration
        start, hook, end = c.clip_start, c.hook, c.clip_end

        def word_at_or_after(t: float) -> Optional[int]:
            i = bisect.bisect_left(starts, t - 0.3)
            while i < len(words) and words[i].end <= t:
                i += 1
            return i if i < len(words) else None

        def word_before(t: float) -> Optional[int]:
            i = bisect.bisect_right(starts, t) - 1
            return i if i >= 0 else None

        # Start on a word boundary (do not begin mid-word)
        i = word_at_or_after(start)
        if i is not None and words[i].start - start < 1.5 and words[i].start <= hook:
            start = words[i].start
        # Prefer to start at the beginning of the sentence containing the start
        segs = transcript.segments_between(start, start + 0.01)
        if segs and 0 < start - segs[0].start <= 3.0 and hook - segs[0].start <= s.max_setup_seconds + 3:
            start = segs[0].start

        # End after the last spoken word; extend to the end of the sentence if close
        j = word_before(end)
        if j is not None and words[j].end > start:
            end = max(end, words[j].end) if end - words[j].end > 1.0 else words[j].end
            k = j
            while (
                k + 1 < len(words)
                and not words[k].word.rstrip().endswith(SENTENCE_END)
                and words[k + 1].start - words[k].end < 0.6
                and words[k + 1].end - end <= 3.0
            ):
                k += 1
            end = max(end, words[k].end)

        start = max(0.0, start - 0.25)  # breathing room
        end = min(duration, end + 0.5)

        # Too long: trim setup first (keep hook + payoff), then trim the tail on a word boundary
        if end - start > s.hard_max_duration:
            new_start = max(start, hook - s.max_setup_seconds)
            i = word_at_or_after(new_start)
            if i is not None and words[i].start <= hook:
                new_start = max(0.0, words[i].start - 0.2)
            start = new_start
        if end - start > s.hard_max_duration:
            limit = start + s.max_clip_duration
            j = word_before(limit)
            while j is not None and j > 0 and words[j].end > limit:
                j -= 1
            # cut at a sentence end if one exists in the last 10 seconds
            k = j
            while k is not None and k > 0 and words[k].end > limit - 10 and not words[k].word.endswith(SENTENCE_END):
                k -= 1
            if k is not None and words[k].end > limit - 10 and words[k].end > hook + 3:
                j = k
            end = (words[j].end + 0.4) if j is not None and words[j].end > hook else limit

        # Too short: extend the end with following speech, then the start backwards
        if end - start < s.min_clip_duration:
            target = s.min_clip_duration
            j = word_at_or_after(end)
            while j is not None and j < len(words) and words[j].end - start < target:
                j += 1
            if j is not None and j < len(words) and words[j].end - start <= s.max_clip_duration:
                end = min(duration, words[j].end + 0.4)
            if end - start < target:
                end = min(duration, start + target)
            if end - start < target:
                start = max(0.0, end - target)

        if end - start < s.min_clip_duration - 1.0:
            c.reject_reason = f"too short ({end - start:.1f}s)"
            return None

        c.clip_start, c.clip_end = round(start, 2), round(end, 2)
        c.hook_start = round(min(max(hook, c.clip_start), c.clip_end), 2)
        return c

    # ------------------------------------------------------------ scoring
    def score(self, c: ClipCandidate) -> float:
        weighted = [(SUBSCORE_WEIGHTS[k], c.subscores[k]) for k in SUBSCORE_WEIGHTS if k in c.subscores]
        final = c.score
        if weighted:
            total_w = sum(w for w, _ in weighted)
            sub = sum(w * v for w, v in weighted) / total_w
            final = 0.6 * c.score + 0.4 * sub
        # mild penalties: outside preferred duration, slow hook
        if not (self.s.preferred_min_duration <= c.duration <= self.s.preferred_max_duration):
            final -= 2.0
        if c.hook - c.clip_start > self.s.max_setup_seconds:
            final -= 3.0
        return round(max(0.0, min(100.0, final)), 1)

    @staticmethod
    def dedupe(cands: List[ClipCandidate]) -> Tuple[List[ClipCandidate], List[ClipCandidate]]:
        kept: List[ClipCandidate] = []
        dropped: List[ClipCandidate] = []
        for c in sorted(cands, key=lambda x: x.final_score, reverse=True):
            dup = next((k for k in kept if is_duplicate(c, k)), None)
            if dup is not None:
                c.reject_reason = f"duplicate of {dup.id}"
                dropped.append(c)
            else:
                kept.append(c)
        return kept, dropped

    # ------------------------------------------------------------ rerank
    def rerank(self, cands: List[ClipCandidate], transcript: Transcript, info: VideoInfo, cache_dir: Path) -> None:
        if not self.llm or len(cands) < 2:
            return
        system, template = load_prompt(self.s.prompts_dir, "highlight_rerank.txt")
        blocks = []
        for c in cands:
            before = transcript.text_between(c.clip_start, c.hook, max_chars=350)
            after = transcript.text_between(c.hook, c.clip_end, max_chars=900)
            blocks.append(
                f"id: {c.id}\nduration: {c.duration:.0f}s | category: {c.category} | first-pass score: {c.score:.0f}\n"
                f"proposed title: {c.title}\neditor's reason: {c.reason}\n"
                f"transcript: {before} >>> {after}"
            )
        user = fill_template(
            template,
            TITLE=info.title,
            CONTEXT=self.s.content_context, AUDIENCE=self.s.target_audience, BRIEF=brief_prompt(self.s),
            NUM_SHORTS=self.s.num_shorts,
            CANDIDATES="\n\n".join(blocks),
        )
        fp = fingerprint([self.llm.label, system, user])
        cache = cache_dir / "rerank.json"
        cached = read_json(cache)
        ids = {c.id for c in cands}

        def validate(data: Any) -> Dict[str, Dict[str, Any]]:
            items = data.get("ranking") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError('expected {"ranking": [...]}')
            out: Dict[str, Dict[str, Any]] = {}
            for it in items:
                if isinstance(it, dict) and str(it.get("id")) in ids:
                    try:
                        sc = float(it.get("score"))
                    except (TypeError, ValueError):
                        continue
                    keep = it.get("keep", True)
                    keep = keep if isinstance(keep, bool) else str(keep).lower() not in ("false", "no", "0")
                    out[str(it["id"])] = {"score": max(0.0, min(100.0, sc)), "keep": keep,
                                          "note": str(it.get("note", ""))[:300]}
            if len(out) < max(1, len(ids) // 2):
                raise ValueError(f"ranking covers only {len(out)} of {len(ids)} candidate ids")
            return out

        if cached and cached.get("fingerprint") == fp:
            result = cached["result"]
            self.log.info("Using cached global re-rank")
        else:
            self.log.info("Global re-rank of %d candidates with %s", len(cands), self.llm.label)
            try:
                result = self.llm.chat_json(system, user, validate=validate, attempts=3,
                                            what="Global re-rank", cancel_event=self.cancel_event)
            except JobCancelled:
                raise
            except Exception as exc:
                self.log.warning("Global re-rank failed (%s); using first-pass scores", exc)
                return
            atomic_write_json(cache, {"fingerprint": fp, "result": result})

        for c in cands:
            r = result.get(c.id)
            if not r:
                continue
            c.rerank_score = r["score"]
            c.rerank_note = r["note"]
            c.final_score = round(0.5 * c.final_score + 0.5 * r["score"], 1)
            if not r["keep"]:
                c.reject_reason = "rejected in global re-rank" + (f": {r['note']}" if r["note"] else "")

    # ------------------------------------------------------------ main
    def rank(
        self,
        candidates: List[ClipCandidate],
        transcript: Transcript,
        info: VideoInfo,
        cache_dir: Path,
    ) -> Tuple[List[ClipCandidate], List[ClipCandidate]]:
        """Return (selected, all_candidates_with_annotations)."""
        s = self.s
        all_cands: List[ClipCandidate] = []
        refined: List[ClipCandidate] = []
        for c in candidates:
            c = c.model_copy(deep=True)
            all_cands.append(c)
            if c.score < s.min_score - 15:
                c.reject_reason = f"low score ({c.score:.0f})"
                continue
            if self.refine(c, transcript) is None:
                continue
            c.final_score = self.score(c)
            refined.append(c)

        kept, _ = self.dedupe(refined)
        pool = [c for c in kept if c.final_score >= s.min_score - 10][: max(s.num_shorts * 3, 12)][:40]
        if s.llm_rerank and self.llm and len(pool) > 1:
            self.rerank(pool, transcript, info, cache_dir)

        ordered = sorted(
            (c for c in kept if not c.reject_reason),
            key=lambda c: c.final_score,
            reverse=True,
        )
        selected: List[ClipCandidate] = []
        for c in ordered:
            if c.final_score < s.min_score:
                c.reject_reason = f"below minimum score ({c.final_score:.0f} < {s.min_score:.0f})"
                continue
            if len(selected) >= s.num_shorts:
                c.reject_reason = "not in top N"
                continue
            if any(is_duplicate(c, k) for k in selected):
                c.reject_reason = "duplicate"
                continue
            selected.append(c)
        for rank, c in enumerate(selected, start=1):
            c.selected = True
            c.rank = rank

        all_cands.sort(key=lambda c: (not c.selected, -(c.final_score or c.score)))
        atomic_write_json(cache_dir / "candidates.json", [c.model_dump() for c in all_cands])
        atomic_write_json(cache_dir / "selected.json", [c.model_dump() for c in selected])
        self.log.info("Selected %d of %d candidates (min score %.0f, target %d)",
                      len(selected), len(candidates), s.min_score, s.num_shorts)
        if len(selected) < s.num_shorts:
            self.log.info("Fewer than %d moments met the quality bar — no filler clips were added", s.num_shorts)
        return selected, all_cands


def load_selected(path: Path) -> List[ClipCandidate]:
    data = read_json(path, [])
    return [ClipCandidate.model_validate(d) for d in data]

