"""Titles, descriptions, hashtags and filenames for every Short + JSON/CSV/Markdown reports."""
from __future__ import annotations

import csv
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Settings
from modules.campaign_brief import brief_prompt, load_brief
from core.llm_client import LLMClient, fill_template, load_prompt
from core.models import ClipCandidate, ShortMetadata, Transcript, VideoInfo
from modules.trend_research import TrendResearcher, derive_query, format_for_prompt, load_source_tags
from core.utils import (
    JobCancelled,
    atomic_write_json,
    atomic_write_text,
    fingerprint,
    fmt_ts,
    read_json,
    slugify,
    strip_unsupported_chars,
)

CATEGORY_TAGS = {
    "funny": "#Funny", "unexpected": "#Unexpected", "shocking": "#Shocking", "reaction": "#Reaction",
    "exciting": "#Epic", "emotional": "#Emotional", "interesting_fact": "#DidYouKnow", "suspense": "#Intense",
    "drama": "#Drama", "discovery": "#EasterEgg", "conversation": "#Podcast", "gameplay": "#Gaming",
}


def clean_tags(tags: Any, extra: Optional[List[str]] = None, max_total: int = 450) -> List[str]:
    """YouTube tags: plain text, deduplicated, no < >, total length (incl. quoting) under 500 chars."""
    if isinstance(tags, str):
        tags = tags.split(",")
    out: List[str] = []
    seen = set()
    total = 0
    for t in list(tags or []) + list(extra or []):
        t = re.sub(r"[<>\"#]", "", strip_unsupported_chars(str(t))).strip()
        t = re.sub(r"\s+", " ", t)
        if not t or len(t) > 60 or t.lower() in seen:
            continue
        cost = len(t) + (2 if " " in t else 0) + 1
        if total + cost > max_total:
            break
        seen.add(t.lower())
        out.append(t)
        total += cost
    return out


def build_description(m: Any, footer: str = "") -> str:
    """Full YouTube description: text, optional footer, then hashtags (first 3 show above the title)."""
    parts = [str(getattr(m, "description", "") or "").strip()]
    if footer.strip():
        parts.append(footer.strip())
    tags = getattr(m, "hashtags", []) or []
    if tags:
        parts.append(" ".join(tags[:15]))  # YouTube ignores ALL hashtags when there are more than 15
    return "\n\n".join(p for p in parts if p)[:4900]


def clean_title(title: str, max_len: int = 90) -> str:
    t = strip_unsupported_chars(title or "").strip().strip('"\'')
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"([!?])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t.isupper() and len(t) > 12:
        t = t.title()
    if len(t) > max_len:
        t = t[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return t


def clean_hashtags(tags: Any, category: str, context: str) -> List[str]:
    if isinstance(tags, str):
        tags = re.split(r"[\s,]+", tags)
    out: List[str] = []
    for t in list(tags or []):
        t = re.sub(r"[^\w]", "", str(t).lstrip("#"))
        if t and len(t) <= 40:
            tag = "#" + t
            if tag.lower() not in {x.lower() for x in out}:
                out.append(tag)
    if "#shorts" not in {x.lower() for x in out}:
        out.insert(0, "#Shorts")
    extra = [CATEGORY_TAGS.get(category, "")]
    if "gta" in (context or "").lower():
        extra.append("#GTA")
    for e in extra:
        if e and e.lower() not in {x.lower() for x in out} and len(out) < 6:
            out.append(e)
    return out[:6]


def enforce_brief(title: str, desc: str, hashtags: List[str], brief: Any) -> Dict[str, Any]:
    """Apply a campaign's hard requirements to the AI's metadata. Returns title/description/hashtags/checks."""
    checks: List[str] = []
    if brief is None:
        return {"title": title, "description": desc, "hashtags": hashtags, "checks": checks}
    req_title = brief.required_title_text.strip()
    if req_title and req_title.lower() not in title.lower():
        room = 100 - len(req_title) - 3
        title = (title[:room].rstrip() + " | " + req_title) if room > 10 else req_title[:100]
        checks.append(f"Title includes “{req_title}”")
    elif req_title:
        checks.append(f"Title includes “{req_title}”")
    extra = []
    if brief.required_description_text and brief.required_description_text.lower() not in desc.lower():
        extra.append(brief.required_description_text.strip())
    if brief.required_description_text:
        checks.append("Required caption text added")
    missing_mentions = [m for m in brief.required_mentions if m.lower() not in desc.lower()]
    if missing_mentions:
        extra.append(" ".join(missing_mentions))
    if brief.required_mentions:
        checks.append("Tagged " + " ".join(brief.required_mentions))
    if extra:
        desc = (desc.rstrip() + "\n\n" + "\n".join(extra)).strip()
    if brief.required_hashtags:
        req = list(brief.required_hashtags)
        rest = [h for h in hashtags if h.lower() not in {r.lower() for r in req}]
        shorts = [h for h in rest if h.lower() == "#shorts"]
        rest = [h for h in rest if h.lower() != "#shorts"]
        hashtags = (shorts + req + rest)[:max(15, len(req) + 1)][:15]
        checks.append("Required hashtags: " + " ".join(req))
    return {"title": title, "description": desc, "hashtags": hashtags, "checks": checks}


def resolve_edit(s: Settings, c: ClipCandidate) -> Dict[str, Any]:
    """Final edit plan for one clip: your explicit setting wins, otherwise the AI's choice, otherwise a default."""
    by: Dict[str, str] = {}

    def pick(name: str, setting: str, ai: Any, valid: tuple, default: Any) -> Any:
        if setting != "auto":
            by[name] = "you"
            return setting
        if ai in valid:
            by[name] = "ai"
            return ai
        by[name] = "default"
        return default

    style = pick("caption_style", s.caption_style, (c.caption_style or "").lower(), ("dynamic", "minimal", "karaoke"),
                 "dynamic")
    layout = pick("layout", s.layout, (c.layout or "").lower(), ("blur_fit", "center_crop", "face_track"), "blur_fit")

    def toggle(name: str, mode: str, ai: Optional[bool], default: bool) -> bool:
        if mode in ("on", "off"):
            by[name] = "you"
            return mode == "on"
        if ai is not None:
            by[name] = "ai"
            return ai
        by[name] = "default"
        return default

    punch = toggle("punch_in", s.punch_in_mode, c.punch_in, s.punch_in)
    zoom = toggle("zoom", s.zoom_mode, c.zoom, s.zoom_effect)
    if s.overlay_text == "off" or not s.title_overlay:
        hook, by["hook_text"] = "", "you"
    elif s.overlay_text == "title":
        hook, by["hook_text"] = "", "you"  # empty = use the title
    else:
        hook = strip_unsupported_chars(c.hook_text or "").strip().strip('"')[:48]
        by["hook_text"] = "ai" if hook else "default"
    return {"caption_style": style, "layout": layout, "punch_in": punch, "zoom": zoom, "hook_text": hook,
            "decided_by": by}


class MetadataGenerator:
    def __init__(
        self,
        settings: Settings,
        llm: Optional[LLMClient] = None,
        logger: Optional[logging.Logger] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.log = logger or logging.getLogger("shorts.metadata")
        self.cancel_event = cancel_event

    def trends(self, info: VideoInfo) -> Optional[Dict[str, Any]]:
        """Trending keywords for this video's topic (None when disabled / no API key / API error)."""
        if not self.s.trend_research or not self.s.youtube_api_key:
            return None
        query = self.s.trend_query or derive_query(info.title, self.s.content_context)
        try:
            return TrendResearcher(self.s, self.log).research(query)
        except Exception as exc:
            self.log.warning("Trend research skipped (%s)", exc)
            return None

    def _metadata_fits(self, clips: List[ClipCandidate], transcript: Transcript, info: VideoInfo,
                       trends: Optional[Dict[str, Any]]) -> bool:
        from core.llm_client import estimate_tokens

        budget = self.s.effective_request_budget
        if not budget:
            return True
        system, template = load_prompt(self.s.prompts_dir, "metadata_generation.txt")
        size = estimate_tokens(system) + estimate_tokens(template) + estimate_tokens(format_for_prompt(trends))
        size += estimate_tokens(brief_prompt(self.s))
        size += sum(estimate_tokens(transcript.text_between(c.clip_start, c.clip_end, max_chars=900)) + 120
                    for c in clips)
        return size <= budget

    def _llm_metadata(self, clips: List[ClipCandidate], transcript: Transcript, info: VideoInfo,
                      cache_dir: Path, trends: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        if not self.llm or not clips:
            return {}
        if len(clips) > 1 and not self._metadata_fits(clips, transcript, info, trends):
            half = len(clips) // 2  # too big for the provider's token limit: write it in batches
            self.log.info("Metadata prompt too large for %s — writing it in batches", self.llm.label)
            return {**self._llm_metadata(clips[:half], transcript, info, cache_dir, trends),
                    **self._llm_metadata(clips[half:], transcript, info, cache_dir, trends)}
        system, template = load_prompt(self.s.prompts_dir, "metadata_generation.txt")
        blocks = []
        for c in clips:
            blocks.append(
                f"id: s{c.rank}\ncategory: {c.category}\ndraft title: {c.title}\non-screen hook: {c.hook_text}\n"
                f"why it works: {c.reason}\nviral trigger: {c.virality}\n"
                f"transcript: {transcript.text_between(c.clip_start, c.clip_end, max_chars=900)}"
            )
        source_tags = load_source_tags(self.s.downloads_dir, info.video_id)
        user = fill_template(template, TITLE=info.title, CONTEXT=self.s.content_context, AUDIENCE=self.s.target_audience, BRIEF=brief_prompt(self.s),
                             SOURCE_TAGS=", ".join(source_tags) or "(none)",
                             LANGUAGE=self.s.youtube_language or self.s.transcription_language or "en",
                             TRENDS=format_for_prompt(trends), SHORTS="\n\n".join(blocks))
        fp = fingerprint([self.llm.label, system, user])
        cache = cache_dir / f"metadata_llm_{fp[:10]}.json"
        cached = read_json(cache)
        if cached and cached.get("fingerprint") == fp:
            return cached["result"]
        ids = {f"s{c.rank}" for c in clips}

        def validate(data: Any) -> Dict[str, Dict[str, Any]]:
            items = data.get("shorts") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError('expected {"shorts": [...]}')
            out = {}
            for it in items:
                if isinstance(it, dict) and str(it.get("id")) in ids and it.get("title"):
                    out[str(it["id"])] = {
                        "title": str(it["title"]),
                        "description": str(it.get("description", "")),
                        "hashtags": it.get("hashtags", []),
                        "tags": it.get("tags", []),
                    }
            if not out:
                raise ValueError("no usable entries")
            return out

        try:
            result = self.llm.chat_json(system, user, validate=validate, attempts=3, what="Metadata generation",
                                        cancel_event=self.cancel_event)
        except JobCancelled:
            raise
        except Exception as exc:
            self.log.warning("Metadata generation via LLM failed (%s); using detection titles", exc)
            return {}
        atomic_write_json(cache, {"fingerprint": fp, "result": result})
        return result

    def generate(self, clips: List[ClipCandidate], transcript: Transcript, info: VideoInfo,
                 cache_dir: Path) -> List[ShortMetadata]:
        trends = self.trends(info) if clips else None
        llm_meta = self._llm_metadata(clips, transcript, info, cache_dir, trends)
        brief = load_brief(self.s)
        topic = (trends or {}).get("query", "")
        metas: List[ShortMetadata] = []
        used_names = set()
        for c in clips:
            m = llm_meta.get(f"s{c.rank}", {})
            title = clean_title(m.get("title") or c.title or f"Best moment #{c.rank}")
            desc = strip_unsupported_chars(m.get("description") or c.description or c.reason).strip()
            hashtags = clean_hashtags(m.get("hashtags") or [], c.category, self.s.content_context)
            yt_tags = clean_tags(m.get("tags") or [], extra=[topic] + [k for k in c.keywords if len(k) > 3])
            enforced = enforce_brief(title, desc, hashtags, brief)
            title, desc, hashtags = enforced["title"], enforced["description"], enforced["hashtags"]
            checks = enforced["checks"]
            if brief is not None:
                if c.id.startswith("g"):
                    checks.insert(0, f"Exact timing from the brief ({c.clip_start:.1f}–{c.clip_end:.1f}s)")
                if self.s.logo_path:
                    checks.append(f"Logo {self.s.logo_position.replace('_', ' ')}")
                elif brief.logo_required:
                    checks.append("Logo required but NO logo file provided")
                if self.s.on_screen_text:
                    checks.append(f"On-screen text “{self.s.on_screen_text}”")
            base = f"{c.rank:02d}_{slugify(title, 50)}"
            name, n = base, 2
            while name in used_names:
                name, n = f"{base}-{n}", n + 1
            used_names.add(name)
            d = resolve_edit(self.s, c)
            metas.append(ShortMetadata(
                rank=c.rank or len(metas) + 1,
                candidate_id=c.id,
                title=title,
                description=desc,
                hashtags=hashtags,
                tags=yt_tags,
                filename=f"{name}.mp4",
                category=c.category,
                score=c.final_score or c.score,
                clip_start=c.clip_start,
                hook_start=c.hook,
                clip_end=c.clip_end,
                duration=round(c.duration, 2),
                reason=c.reason,
                keywords=c.keywords,
                caption_style=d["caption_style"],
                hook_text=d["hook_text"],
                layout=d["layout"],
                punch_in=d["punch_in"],
                zoom=d["zoom"],
                edit_notes=c.edit_notes,
                virality=c.virality,
                decided_by=d["decided_by"],
                requirements=checks,
            ))
        return metas


# ---------------------------------------------------------------- reports
def write_metadata_files(out_dir: Path, metas: List[ShortMetadata]) -> None:
    atomic_write_json(out_dir / "metadata.json", [m.model_dump() for m in metas])
    path = out_dir / "metadata.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel opens UTF-8 correctly
        w = csv.writer(fh)
        w.writerow(["rank", "status", "filename", "title", "description", "hashtags", "tags", "category", "score",
                    "clip_start", "hook_start", "clip_end", "duration", "caption_style", "reason", "output_path"])
        for m in metas:
            w.writerow([m.rank, m.status, m.filename, m.title, m.description, " ".join(m.hashtags),
                        ", ".join(m.tags), m.category,
                        m.score, m.clip_start, m.hook_start, m.clip_end, m.duration, m.caption_style, m.reason,
                        m.output_path])
    tmp.replace(path)


def write_report(out_dir: Path, info: VideoInfo, transcript: Transcript, candidates: List[ClipCandidate],
                 metas: List[ShortMetadata], settings: Settings, job_id: str, llm_label: str) -> Path:
    report = {
        "job_id": job_id,
        "video": info.model_dump(),
        "transcript": {"segments": len(transcript.transcript), "words": len(transcript.all_words()),
                       "language": transcript.language, "model": transcript.model},
        "llm": llm_label,
        "settings": settings.public_dict(),
        "summary": {
            "candidates": len(candidates),
            "selected": sum(1 for c in candidates if c.selected),
            "exported": sum(1 for m in metas if m.status == "exported"),
            "failed": sum(1 for m in metas if m.status == "failed"),
        },
        "shorts": [m.model_dump() for m in metas],
        "candidates": [c.model_dump() for c in candidates],
    }
    atomic_write_json(out_dir / "report.json", report)

    lines = [
        f"# Shorts report — {info.title}",
        "",
        f"- Source: {info.webpage_url or info.source}",
        f"- Duration: {fmt_ts(info.duration, False)} | Resolution: {info.resolution}",
        f"- Transcription: {transcript.model} ({transcript.language or 'auto'}) | Highlight model: {llm_label}",
        f"- Candidates: {report['summary']['candidates']} | Selected: {report['summary']['selected']} | "
        f"Exported: {report['summary']['exported']} | Failed: {report['summary']['failed']}",
        "",
        "## Exported Shorts",
        "",
        "| # | Score | Time | Len | Category | Title | Status | File |",
        "|---|------:|------|----:|----------|-------|--------|------|",
    ]
    for m in metas:
        lines.append(f"| {m.rank} | {m.score:.0f} | {fmt_ts(m.clip_start, False)}–{fmt_ts(m.clip_end, False)} | "
                     f"{m.duration:.0f}s | {m.category} | {m.title.replace('|', '/')} | {m.status} | {m.filename} |")
    lines += ["", "## All candidates", "",
              "| Selected | Score (final / first pass) | Time | Category | Title | Reason / rejection |",
              "|---|---|---|---|---|---|"]
    for c in candidates:
        why = (c.reject_reason or c.reason).replace("|", "/")
        lines.append(f"| {'✅ #' + str(c.rank) if c.selected else ''} | {c.final_score:.0f} / {c.score:.0f} | "
                     f"{fmt_ts(c.clip_start, False)}–{fmt_ts(c.clip_end, False)} | {c.category} | "
                     f"{c.title.replace('|', '/')} | {why} |")
    atomic_write_text(out_dir / "report.md", "\n".join(lines) + "\n")
    return out_dir / "report.json"
