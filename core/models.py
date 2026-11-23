"""Pydantic data models shared across the pipeline."""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from core.utils import fmt_ts, parse_timestamp


# ============================================================== video
class VideoInfo(BaseModel):
    video_id: str
    source: str
    title: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True
    uploader: str = ""
    upload_date: str = ""
    webpage_url: str = ""
    source_path: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "unknown"


# ============================================================== transcript
class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: List[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    video_id: str
    duration: float
    language: Optional[str] = None
    model: str = ""
    fingerprint: str = ""
    transcript: List[Segment] = Field(default_factory=list)

    _words: Optional[List[Word]] = PrivateAttr(default=None)
    _word_starts: Optional[List[float]] = PrivateAttr(default=None)
    _seg_starts: Optional[List[float]] = PrivateAttr(default=None)

    # -------------------------------------------------------------- lookups
    def all_words(self) -> List[Word]:
        if self._words is None:
            words = [w for seg in self.transcript for w in seg.words]
            words.sort(key=lambda w: w.start)
            self._words = words
            self._word_starts = [w.start for w in words]
        return self._words

    def words_between(self, start: float, end: float) -> List[Word]:
        words = self.all_words()
        assert self._word_starts is not None
        i = max(0, bisect.bisect_left(self._word_starts, start - 2.0))
        out = []
        while i < len(words) and words[i].start <= end:
            w = words[i]
            if start - 0.05 <= w.mid <= end + 0.05:
                out.append(w)
            i += 1
        return out

    def segments_between(self, start: float, end: float) -> List[Segment]:
        if self._seg_starts is None:
            self._seg_starts = [s.start for s in self.transcript]
        i = max(0, bisect.bisect_right(self._seg_starts, start) - 3)
        out = []
        while i < len(self.transcript) and self.transcript[i].start < end:
            seg = self.transcript[i]
            if seg.end > start:
                out.append(seg)
            i += 1
        return out

    def text_between(self, start: float, end: float, max_chars: int = 0) -> str:
        words = self.words_between(start, end)
        segs = self.segments_between(start, end)
        if segs and (not words or len(segs) <= 2):
            text = " ".join(s.text.strip() for s in segs)
        else:
            text = " ".join(w.word for w in words)
        text = " ".join(text.split())
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        return text

    def to_txt(self, title: str = "") -> str:
        lines = []
        if title:
            lines += [f"# {title}", f"# video_id: {self.video_id}  duration: {fmt_ts(self.duration)}", ""]
        for seg in self.transcript:
            lines.append(f"[{fmt_ts(seg.start)} -> {fmt_ts(seg.end)}] {seg.text.strip()}")
        return "\n".join(lines) + "\n"


# ============================================================== highlights
SUBSCORE_KEYS = ("hook", "entertainment", "originality", "emotion", "clarity", "retention", "standalone")


class ClipCandidate(BaseModel):
    """A highlight moment. Produced by the LLM, refined by the ranker."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    clip_start: float
    hook_start: Optional[float] = None
    clip_end: float
    score: float = 0.0
    subscores: Dict[str, float] = Field(default_factory=dict)
    category: str = "other"
    reason: str = ""
    title: str = ""
    description: str = ""
    caption_style: str = ""
    keywords: List[str] = Field(default_factory=list)
    # AI editing decisions
    hook_text: str = ""  # short on-screen hook line
    layout: str = ""
    punch_in: Optional[bool] = None
    zoom: Optional[bool] = None
    edit_notes: str = ""
    virality: str = ""  # why this will travel (share / comment / rewatch trigger)

    window_index: int = -1
    final_score: float = 0.0
    rerank_score: Optional[float] = None
    rerank_note: str = ""
    selected: bool = False
    rank: Optional[int] = None
    reject_reason: str = ""

    @field_validator("clip_start", "clip_end", mode="before")
    @classmethod
    def _parse_ts(cls, v: Any) -> float:
        return parse_timestamp(v)

    @field_validator("hook_start", mode="before")
    @classmethod
    def _parse_hook(cls, v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        return parse_timestamp(v)

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if 0 < v <= 1.0:  # model answered on a 0-1 scale
            v *= 100
        elif 1.0 < v <= 10.0:  # 0-10 scale
            v *= 10
        return max(0.0, min(100.0, v))

    @field_validator("subscores", mode="before")
    @classmethod
    def _subscores(cls, v: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(v, dict):
            for k, val in v.items():
                try:
                    out[str(k).lower().strip()] = float(val)
                except (TypeError, ValueError):
                    continue
        if out and max(out.values()) <= 10:  # model used a 0-10 scale
            out = {k: x * 10 for k, x in out.items()}
        return {k: max(0.0, min(100.0, x)) for k, x in out.items()}

    @field_validator("keywords", mode="before")
    @classmethod
    def _keywords(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.split(",")
        return [str(x).strip() for x in v if str(x).strip()][:8]

    @field_validator("category", "reason", "title", "description", "caption_style", "hook_text", "layout",
                     "edit_notes", "virality", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()

    @field_validator("punch_in", "zoom", mode="before")
    @classmethod
    def _opt_bool(cls, v: Any) -> Optional[bool]:
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        t = str(v).strip().lower()
        if t in ("true", "yes", "on", "1"):
            return True
        if t in ("false", "no", "off", "0"):
            return False
        return None

    @model_validator(mode="after")
    def _check_times(self) -> "ClipCandidate":
        if self.clip_end < self.clip_start:
            self.clip_start, self.clip_end = self.clip_end, self.clip_start
        if self.clip_end - self.clip_start < 1.0:
            raise ValueError("clip_end must be after clip_start")
        if self.hook_start is None or not (self.clip_start <= self.hook_start <= self.clip_end):
            self.hook_start = self.clip_start if self.hook_start is None else min(
                max(self.hook_start, self.clip_start), self.clip_end
            )
        self.category = (self.category or "other").lower()
        return self

    @property
    def duration(self) -> float:
        return self.clip_end - self.clip_start

    @property
    def hook(self) -> float:
        return self.hook_start if self.hook_start is not None else self.clip_start


# ============================================================== output
class ShortMetadata(BaseModel):
    rank: int
    candidate_id: str = ""
    title: str
    description: str = ""
    hashtags: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)  # YouTube "tags" field (not shown to viewers)
    filename: str
    category: str = "other"
    score: float = 0.0
    clip_start: float
    hook_start: float
    clip_end: float
    duration: float
    reason: str = ""
    keywords: List[str] = Field(default_factory=list)
    caption_style: str = "dynamic"
    hook_text: str = ""  # on-screen text (AI hook line)
    layout: str = ""  # resolved layout for this Short
    punch_in: Optional[bool] = None
    zoom: Optional[bool] = None
    edit_notes: str = ""
    virality: str = ""
    decided_by: Dict[str, str] = Field(default_factory=dict)  # setting -> "ai" | "you" | "default"
    requirements: List[str] = Field(default_factory=list)  # campaign requirements applied to this clip
    output_path: str = ""
    thumbnail_path: str = ""
    status: str = "pending"  # pending | exported | failed
    render_mode: str = ""
    render_fingerprint: str = ""
    qc: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""
