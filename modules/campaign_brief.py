"""Clipping-campaign briefs: free-text requirements → structured spec → pipeline settings.

Flow
  1. ``BriefParser.parse(text)`` — the LLM extracts links, exact timings, clip count,
     durations, hashtags, mentions, required text, logo rules, caption/music rules,
     dos/don'ts. A regex fallback still finds links, timings, hashtags and mentions.
  2. ``save_campaign`` stores brief.txt / brief.json / logo in campaigns/<slug>/.
  3. ``campaign_overrides`` maps the spec onto Settings for each job.
  4. The pipeline reads ``Settings.campaign_json``:
       * exact segments in the brief → used as-is (AI only directs hook text/title/edit)
       * no segments → hook-based highlight detection guided by the brief
     and the metadata step enforces required hashtags / mentions / text.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import Settings
from core.utils import atomic_write_json, atomic_write_text, parse_timestamp, slugify

LOGO_POSITIONS = ("top_right", "top_left", "top_center", "bottom_right", "bottom_left")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
TS = r"(\d{1,2}:\d{2}(?::\d{2})?)"
RANGE_RE = re.compile(TS + r"\s*(?:-|–|—|to|until)\s*" + TS, re.I)


class BriefSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = ""
    start: float
    end: float
    note: str = ""

    @field_validator("start", "end", mode="before")
    @classmethod
    def _ts(cls, v: Any) -> float:
        return parse_timestamp(v)

    @field_validator("url", "note", mode="before")
    @classmethod
    def _s(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()


class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    brand: str = ""
    goal: str = ""
    content_focus: str = ""
    audience: str = ""
    source_urls: List[str] = Field(default_factory=list)
    search_topic: str = ""
    segments: List[BriefSegment] = Field(default_factory=list)
    num_clips: Optional[int] = None
    extra_clips: int = 0  # AI-found clips wanted IN ADDITION to the exact segments
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    required_hashtags: List[str] = Field(default_factory=list)
    required_mentions: List[str] = Field(default_factory=list)
    required_title_text: str = ""
    required_description_text: str = ""
    on_screen_text: str = ""
    logo_required: bool = False
    logo_position: str = "top_right"
    logo_url: str = ""
    captions_required: Optional[bool] = None
    caption_style: str = ""
    music_allowed: Optional[bool] = None
    language: str = ""
    platforms: List[str] = Field(default_factory=list)
    dos: List[str] = Field(default_factory=list)
    donts: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    raw_text: str = ""

    @field_validator("name", "brand", "goal", "content_focus", "audience", "search_topic", "required_title_text",
                     "required_description_text", "on_screen_text", "logo_url", "caption_style", "language",
                     mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()

    @field_validator("source_urls", "platforms", "dos", "donts", "missing_info", mode="before")
    @classmethod
    def _list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [x for x in re.split(r"[\n,]+", v)]
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("segments", mode="before")
    @classmethod
    def _segments(cls, v: Any) -> List[Any]:
        out = []
        for item in v or []:
            try:
                seg = BriefSegment.model_validate(item)
                if seg.end > seg.start:
                    out.append(seg)
            except Exception:
                continue
        return out

    @field_validator("required_hashtags", mode="before")
    @classmethod
    def _hashtags(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = re.split(r"[\s,]+", v)
        out: List[str] = []
        for t in v or []:
            t = "#" + re.sub(r"[^\w]", "", str(t).lstrip("#"))
            if len(t) > 1 and t.lower() not in {x.lower() for x in out}:
                out.append(t)
        return out

    @field_validator("required_mentions", mode="before")
    @classmethod
    def _mentions(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = re.split(r"[\s,]+", v)
        out: List[str] = []
        for t in v or []:
            t = "@" + re.sub(r"[^\w.]", "", str(t).lstrip("@")).strip(".")
            if len(t) > 1 and t.lower() not in {x.lower() for x in out}:
                out.append(t)
        return out

    @field_validator("num_clips", mode="before")
    @classmethod
    def _int(cls, v: Any) -> Optional[int]:
        try:
            n = int(float(v))
            return n if 0 < n <= 50 else None
        except (TypeError, ValueError):
            return None

    @field_validator("extra_clips", mode="before")
    @classmethod
    def _extra(cls, v: Any) -> int:
        try:
            return max(0, min(20, int(float(v))))
        except (TypeError, ValueError):
            return 0

    @field_validator("min_duration", "max_duration", mode="before")
    @classmethod
    def _dur(cls, v: Any) -> Optional[float]:
        if v in (None, "", 0):
            return None
        try:
            d = parse_timestamp(v)
            return d if 1 <= d <= 600 else None
        except Exception:
            return None

    @field_validator("logo_position", mode="before")
    @classmethod
    def _pos(cls, v: Any) -> str:
        v = str(v or "top_right").strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in LOGO_POSITIONS else "top_right"

    @field_validator("captions_required", "music_allowed", mode="before")
    @classmethod
    def _opt_bool(cls, v: Any) -> Optional[bool]:
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        t = str(v).strip().lower()
        return True if t in ("true", "yes", "1") else False if t in ("false", "no", "0") else None

    # -------------------------------------------------------------- helpers
    def segments_for(self, video_id: str, source: str) -> List[BriefSegment]:
        """Segments that belong to this source (blank url = applies when the brief has one video)."""
        from modules.downloader import parse_youtube_url

        out = []
        for seg in self.segments:
            if not seg.url:
                if len(self.source_urls) <= 1:
                    out.append(seg)
                continue
            vid = parse_youtube_url(seg.url)
            if (vid and vid == video_id) or seg.url.strip() == source.strip():
                out.append(seg)
        return out

    def requirement_lines(self) -> List[str]:
        lines = []
        if self.goal:
            lines.append(f"Goal: {self.goal}")
        if self.content_focus:
            lines.append(f"Clip this kind of content: {self.content_focus}")
        if self.audience:
            lines.append(f"Audience: {self.audience}")
        if self.num_clips:
            lines.append(f"Number of clips wanted: {self.num_clips}")
        if self.segments:
            lines.append(f"Exact segments: {len(self.segments)}" +
                         (f" + up to {self.extra_clips} more found by AI" if self.extra_clips else " (only these)"))
        if self.min_duration or self.max_duration:
            lines.append(f"Clip length: {self.min_duration or '?'}–{self.max_duration or '?'} seconds")
        if self.required_hashtags:
            lines.append("Required hashtags: " + " ".join(self.required_hashtags))
        if self.required_mentions:
            lines.append("Must tag: " + " ".join(self.required_mentions))
        if self.required_title_text:
            lines.append(f"Every title must include: {self.required_title_text}")
        if self.required_description_text:
            lines.append(f"Post caption must include: {self.required_description_text}")
        if self.on_screen_text:
            lines.append(f"On-screen text required: {self.on_screen_text}")
        if self.logo_required:
            lines.append(f"Brand logo shown {self.logo_position.replace('_', ' ')} (added automatically)")
        if self.captions_required is not None:
            lines.append("Subtitles on video: " + ("required" if self.captions_required else "not allowed"))
        if self.music_allowed is False:
            lines.append("No added music")
        if self.platforms:
            lines.append("Platforms: " + ", ".join(self.platforms))
        lines += [f"Do: {d}" for d in self.dos]
        lines += [f"Don't: {d}" for d in self.donts]
        return lines

    def for_prompt(self) -> str:
        lines = self.requirement_lines()
        if not lines:
            return "(no campaign brief — use general best practice for virality)"
        head = f"Campaign: {self.name or 'unnamed'}" + (f" by {self.brand}" if self.brand else "")
        return head + "\nThese requirements OVERRIDE general preferences. Follow them exactly.\n- " + "\n- ".join(lines)


def load_brief(settings: Settings) -> Optional[CampaignBrief]:
    if not settings.campaign_json:
        return None
    try:
        return CampaignBrief.model_validate(json.loads(settings.campaign_json))
    except Exception:
        return None


def brief_prompt(settings: Settings) -> str:
    b = load_brief(settings)
    return b.for_prompt() if b else "(no campaign brief — use general best practice for virality)"


# ------------------------------------------------------------------ parsing
def regex_extract(text: str) -> Dict[str, Any]:
    """Fallback / safety net: links, timings, hashtags and mentions straight from the text."""
    urls = list(dict.fromkeys(u.rstrip(".,;") for u in URL_RE.findall(text)))
    video_urls = [u for u in urls if not Path(u.split("?")[0]).suffix.lower() in IMAGE_EXTS]
    segments = []
    for line in text.splitlines():
        line_urls = URL_RE.findall(line)
        for a, b in RANGE_RE.findall(line):
            segments.append({"url": line_urls[0] if line_urls else "", "start": a, "end": b})
    no_urls = URL_RE.sub(" ", text)
    hashtags = re.findall(r"#\w+", no_urls)
    mentions = re.findall(r"(?<![\w.])@\w(?:[\w.]{0,28}\w)?", no_urls)
    logo = [u for u in urls if Path(u.split("?")[0]).suffix.lower() in IMAGE_EXTS]
    return {"source_urls": video_urls, "segments": segments, "required_hashtags": hashtags,
            "required_mentions": mentions, "logo_url": logo[0] if logo else ""}


class BriefParser:
    def __init__(self, settings: Settings, llm: Any = None, logger: Optional[logging.Logger] = None) -> None:
        self.s = settings
        self.llm = llm
        self.log = logger or logging.getLogger("shorts.campaign")

    def parse(self, text: str) -> CampaignBrief:
        text = (text or "").strip()
        if not text:
            raise ValueError("The campaign brief is empty.")
        rx = regex_extract(text)
        data: Dict[str, Any] = {}
        if self.llm is not None:
            from core.llm_client import fill_template, load_prompt

            system, template = load_prompt(self.s.prompts_dir, "campaign_brief.txt")

            def validate(d: Any) -> Dict[str, Any]:
                if not isinstance(d, dict):
                    raise ValueError("expected a JSON object")
                CampaignBrief.model_validate(d)
                return d

            try:
                data = self.llm.chat_json(system, fill_template(template, BRIEF=text[:20000]), validate=validate,
                                          attempts=3, what="Campaign brief analysis")
            except Exception as exc:
                self.log.warning("AI brief analysis failed (%s); using pattern extraction only", exc)
                data = {"missing_info": [f"AI analysis failed ({exc}); only links, timings and hashtags were read"]}
        else:
            data = {"missing_info": ["No AI provider configured; only links, timings and hashtags were read"]}

        # merge: regex guarantees nothing explicit is lost
        for key in ("source_urls", "required_hashtags", "required_mentions"):
            merged = list(data.get(key) or [])
            low = {str(x).lower() for x in merged}
            merged += [x for x in rx[key] if x.lower() not in low]
            data[key] = merged
        if not data.get("segments") and rx["segments"]:
            data["segments"] = rx["segments"]
        if not data.get("logo_url") and rx["logo_url"]:
            data["logo_url"] = rx["logo_url"]
        brief = CampaignBrief.model_validate({**data, "raw_text": text})
        # only keep links that are actually videos we can process
        from modules.downloader import parse_youtube_url

        unsupported = [u for u in brief.source_urls if not parse_youtube_url(u)]
        brief.source_urls = [u for u in brief.source_urls if parse_youtube_url(u)]
        if unsupported:
            brief.missing_info.append("Links that are not YouTube videos were ignored (download them and use "
                                      "Local file): " + ", ".join(unsupported[:5]))
        if not brief.source_urls and not brief.search_topic:
            brief.missing_info.append("No video link found — add one in the links box.")
        return brief


# ------------------------------------------------------------------ storage
def fetch_logo(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a direct image link (png/jpg/webp). Returns None when it is not an image."""
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200 or not ctype.startswith("image/") or "svg" in ctype:
            return None
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(ctype.split(";")[0], ".png")
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"logo{ext}"
        path.write_bytes(r.content)
        return path
    except httpx.HTTPError:
        return None


def save_campaign(settings: Settings, brief: CampaignBrief, logo_bytes: Optional[bytes] = None,
                  logo_name: str = "") -> Dict[str, Any]:
    """Persist the brief and logo. Returns {"dir", "logo_path", "slug"}."""
    slug = f"{slugify(brief.name or brief.brand or 'campaign', 40)}-{datetime.now():%Y%m%d-%H%M%S}"
    folder = settings.data_dir / "campaigns" / slug
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_text(folder / "brief.txt", brief.raw_text)
    logo_path: Optional[Path] = None
    if logo_bytes:
        ext = Path(logo_name).suffix.lower() if Path(logo_name).suffix.lower() in IMAGE_EXTS else ".png"
        logo_path = folder / f"logo{ext}"
        logo_path.write_bytes(logo_bytes)
    elif brief.logo_url:
        logo_path = fetch_logo(brief.logo_url, folder)
        if logo_path is None:
            brief.missing_info.append("Logo link is not a direct image — upload the logo file instead.")
    if brief.logo_required and logo_path is None:
        brief.missing_info.append("The brief requires a logo but no logo file was provided.")
    atomic_write_json(folder / "brief.json", brief.model_dump())
    return {"dir": str(folder), "logo_path": str(logo_path) if logo_path else "", "slug": slug}


def campaign_overrides(settings: Settings, brief: CampaignBrief, logo_path: str = "",
                       slug: str = "") -> Dict[str, Any]:
    """Settings a campaign imposes on its jobs. Brief requirements override your defaults."""
    o: Dict[str, Any] = {"campaign_json": brief.model_dump_json(exclude={"raw_text"}), "campaign_slug": slug}
    focus = " — ".join(x for x in (brief.content_focus, brief.goal) if x)
    if focus:
        o["content_context"] = f"{focus} ({brief.brand})" if brief.brand else focus
    if brief.audience:
        o["target_audience"] = brief.audience
    if brief.segments and brief.extra_clips:
        o["num_shorts"] = brief.extra_clips
        o["min_score"] = min(settings.min_score, 50.0)
    elif brief.num_clips:
        o["num_shorts"] = brief.num_clips
        o["min_score"] = min(settings.min_score, 50.0)  # deliver the requested count when the moments exist
    lo = brief.min_duration or settings.min_clip_duration
    hi = brief.max_duration or settings.max_clip_duration
    if brief.min_duration or brief.max_duration:
        lo, hi = min(lo, hi - 1), max(hi, lo + 1)
        o.update(min_clip_duration=lo, max_clip_duration=hi, duration_tolerance=0.0 if brief.max_duration else 0.15,
                 preferred_min_duration=lo + (hi - lo) * 0.25, preferred_max_duration=lo + (hi - lo) * 0.85)
    if brief.captions_required is not None:
        o["captions_enabled"] = brief.captions_required
    if brief.caption_style in ("dynamic", "minimal", "karaoke"):
        o["caption_style"] = brief.caption_style
    if brief.music_allowed is False:
        o["background_music"] = ""
    if brief.language:
        o["transcription_language"] = brief.language[:2].lower()
        o["youtube_language"] = brief.language[:2].lower()
    if brief.on_screen_text:
        o["on_screen_text"] = brief.on_screen_text
    if logo_path:
        o["logo_path"] = logo_path
        o["logo_position"] = brief.logo_position
    if brief.search_topic or brief.content_focus:
        o["trend_query"] = settings.trend_query or brief.search_topic or brief.content_focus[:60]
    return o
