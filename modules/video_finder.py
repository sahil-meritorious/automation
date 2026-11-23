"""Topic → source videos ("GTA 5 vs GTA 6 comparison" → the best long video to clip).

Only searches videos you are allowed to reuse:

* ``any_public``       — TESTING: any public video. Jobs are marked private_only, so
  their Shorts can only ever be uploaded as private (never public/unlisted)
* ``creative_commons`` — YouTube videos licensed CC BY (reuse allowed with credit;
  a credit line is added to every description automatically)
* ``my_channel``       — videos on your own connected channel (OAuth)
* ``allowed_channels`` — channels you have permission from (ALLOWED_CHANNEL_IDS)

Candidates are filtered by length and licence, then the LLM picks the ones with
the most viral Shorts potential for the target audience (heuristic fallback).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import Settings
from core.llm_client import LLMClient
from modules.trend_research import TrendError, TrendResearcher

SOURCE_LABELS = {
    "any_public": "Any public video (testing: uploads forced private)",
    "creative_commons": "Creative Commons videos (reuse allowed with credit)",
    "my_channel": "My own channel",
    "allowed_channels": "Channels I have permission from",
}

PICK_SYSTEM = """You choose source videos for a YouTube Shorts channel whose goal is VIRALITY.
Given a topic, the target audience and candidate long videos, pick the ones most likely to contain
several scroll-stopping, shareable moments about that topic (strong reactions, comparisons, discoveries,
easter eggs, chaos, commentary). Prefer videos that are clearly about the topic, have commentary or
dialogue (the clips are found from speech), are recent and popular. Avoid compilations of other people's
clips, music-only videos, livestream waiting screens and unrelated content.
IMPORTANT: reject videos that look like RE-UPLOADS of other creators' content (titles naming other
streamers/YouTubers, "reacts", "reaction", "compilation", "best clips", "ft.", stream highlights posted by
a different channel). Their Creative Commons label is usually invalid because the uploader does not own
the footage. Prefer original content where the channel itself is the creator.
Reply with ONE JSON object only: {"picks": [{"id": "<video id>", "reason": "short reason"}]} — best first."""


@dataclass
class FoundVideo:
    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str
    duration: float
    views: int
    published: str
    license: str
    description: str = ""
    reason: str = ""

    def credit(self) -> str:
        if self.license == "creativeCommon":
            return f'Source: "{self.title}" by {self.channel} — {self.url} (CC BY)'
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinderError(RuntimeError):
    pass


REUPLOAD_RE = re.compile(
    r"\b(reacts?|reaction|compilation|best clips|clips|highlights|ft\.|feat\.|re-?upload|asmongold|xqc|"
    r"kai cenat|ishowspeed|mrbeast|pewdiepie|summit1g|shroud|ninja|pokimane|hasan)\b", re.I)


def looks_like_reupload(v: "FoundVideo") -> bool:
    """Titles that name other creators or are compilations are rarely owned by the uploader."""
    return bool(REUPLOAD_RE.search(v.title)) and v.channel.lower() not in v.title.lower()


def iso_duration(value: str) -> float:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not m:
        return 0.0
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


class VideoFinder:
    def __init__(self, settings: Settings, llm: Optional[LLMClient] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.s = settings
        self.llm = llm
        self.log = logger or logging.getLogger("shorts.finder")
        self.api = TrendResearcher(settings, self.log)

    # ------------------------------------------------------------------ search
    def _search_api_key(self, topic: str, **extra: Any) -> List[str]:
        data = self.api._get("search", part="snippet", q=topic, type="video", maxResults=50,
                             relevanceLanguage=self.s.youtube_language or "en", safeSearch="none", **extra)
        return [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]

    def allowed_channels(self) -> List[str]:
        """ALLOWED_CHANNEL_IDS entries as channel IDs; @handles are resolved via channels.list (1 unit each)."""
        out: List[str] = []
        for raw in [c.strip() for c in self.s.allowed_channel_ids.split(",") if c.strip()]:
            if raw.startswith("UC") and len(raw) == 24:
                out.append(raw)
                continue
            handle = raw.rstrip("/").split("/")[-1]
            handle = handle if handle.startswith("@") else "@" + handle
            try:
                items = self.api._get("channels", part="id", forHandle=handle).get("items", [])
            except TrendError as exc:
                self.log.warning("Could not resolve channel %s: %s", raw, exc)
                continue
            if items:
                out.append(items[0]["id"])
            else:
                self.log.warning("Channel %s not found — check ALLOWED_CHANNEL_IDS", raw)
        return out

    def _search_my_channel(self, topic: str) -> List[str]:
        from modules.youtube_uploader import youtube_service

        yt = youtube_service(self.s)
        data = yt.search().list(part="snippet", forMine=True, type="video", q=topic, maxResults=50).execute()
        return [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]

    def _details(self, ids: List[str]) -> List[FoundVideo]:
        out: List[FoundVideo] = []
        for i in range(0, len(ids), 50):
            data = self.api._get("videos", part="snippet,contentDetails,statistics,status",
                                 id=",".join(ids[i:i + 50]), maxResults=50)
            for v in data.get("items", []):
                sn, cd, st = v.get("snippet", {}), v.get("contentDetails", {}), v.get("status", {})
                if sn.get("liveBroadcastContent") in ("live", "upcoming"):
                    continue
                out.append(FoundVideo(
                    video_id=v["id"], url=f"https://www.youtube.com/watch?v={v['id']}",
                    title=sn.get("title", ""), channel=sn.get("channelTitle", ""), channel_id=sn.get("channelId", ""),
                    duration=iso_duration(cd.get("duration", "")),
                    views=int(v.get("statistics", {}).get("viewCount", 0) or 0),
                    published=sn.get("publishedAt", ""), license=st.get("license", "youtube"),
                    description=(sn.get("description") or "")[:300],
                ))
        return out

    def candidates(self, topic: str) -> List[FoundVideo]:
        src = self.s.topic_source
        if src != "my_channel" and not self.s.youtube_api_key:
            raise FinderError("Topic search needs YOUTUBE_API_KEY (YouTube Data API v3 key).")
        try:
            if src == "creative_commons":
                ids = self._search_api_key(topic, videoLicense="creativeCommon", order="relevance")
            elif src == "any_public":
                ids = self._search_api_key(topic, order="relevance")
            elif src == "my_channel":
                ids = self._search_my_channel(topic)
            else:
                channels = self.allowed_channels()
                if not channels:
                    raise FinderError("Set ALLOWED_CHANNEL_IDS in .env (comma-separated channel IDs or @handles "
                                      "you have permission from).")
                ids = []
                for ch in channels[:10]:
                    ids += self._search_api_key(topic, channelId=ch, order="relevance")
        except TrendError as exc:
            raise FinderError(str(exc)) from exc
        if not ids:
            return []
        if src == "my_channel" and not self.s.youtube_api_key:
            raise FinderError("Reading video details needs YOUTUBE_API_KEY as well.")
        vids = self._details(list(dict.fromkeys(ids)))
        lo, hi = self.s.topic_min_minutes * 60, self.s.topic_max_minutes * 60
        vids = [v for v in vids if lo <= v.duration <= hi]
        if src == "creative_commons":
            vids = [v for v in vids if v.license == "creativeCommon"]  # double-check the licence
            dropped = [v for v in vids if looks_like_reupload(v)]
            vids = [v for v in vids if not looks_like_reupload(v)]
            for v in dropped:
                self.log.info("Skipped likely re-upload (CC label not trustworthy): %s — %s", v.title, v.channel)
        return vids

    # ------------------------------------------------------------------ choose
    @staticmethod
    def _heuristic(topic: str, v: FoundVideo) -> float:
        words = {w for w in re.findall(r"\w+", topic.lower()) if len(w) > 1}
        title = set(re.findall(r"\w+", v.title.lower()))
        overlap = len(words & title) / max(1, len(words))
        try:
            age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(v.published.replace("Z", "+00:00"))).days
        except ValueError:
            age_days = 365
        freshness = 1 / (1 + age_days / 90)
        return overlap * 3 + math.log10(1 + v.views) * 0.6 + freshness

    def find(self, topic: str, count: int = 1, exclude_ids: Optional[List[str]] = None) -> List[FoundVideo]:
        exclude = set(exclude_ids or [])
        vids = [v for v in self.candidates(topic) if v.video_id not in exclude]
        if not vids:
            raise FinderError(
                f"No suitable videos found for '{topic}' in {SOURCE_LABELS[self.s.topic_source]} "
                f"({self.s.topic_min_minutes:.0f}–{self.s.topic_max_minutes:.0f} min). Try a broader topic.")
        vids.sort(key=lambda v: self._heuristic(topic, v), reverse=True)
        shortlist = vids[:20]
        picks = self._llm_pick(topic, shortlist, count) if self.llm else []
        chosen = picks or shortlist[:count]
        for v in chosen:
            self.log.info("Topic '%s' → %s (%s, %.0f min, %s views) %s", topic, v.title, v.channel,
                          v.duration / 60, f"{v.views:,}", v.reason)
        return chosen[:count]

    def _llm_pick(self, topic: str, vids: List[FoundVideo], count: int) -> List[FoundVideo]:
        by_id = {v.video_id: v for v in vids}
        lines = [f"id: {v.video_id}\ntitle: {v.title}\nchannel: {v.channel}\nlength: {v.duration / 60:.0f} min\n"
                 f"views: {v.views:,}\npublished: {v.published[:10]}\ndescription: {v.description[:200]}" for v in vids]
        user = (f"Topic: {topic}\nTarget audience: {self.s.target_audience}\nPick the best {count} video(s).\n\n"
                + "\n\n".join(lines))

        def validate(data: Any) -> List[FoundVideo]:
            items = data.get("picks") if isinstance(data, dict) else data
            out = []
            for it in items or []:
                vid = by_id.get(str(it.get("id", "")).strip()) if isinstance(it, dict) else None
                if vid and vid not in out:
                    vid.reason = str(it.get("reason", ""))[:200]
                    out.append(vid)
            if not out:
                raise ValueError("no valid ids")
            return out

        try:
            return self.llm.chat_json(PICK_SYSTEM, user, validate=validate, attempts=2, what="Source video pick")
        except Exception as exc:
            self.log.warning("AI video pick failed (%s); using ranking by relevance and views", exc)
            return []


def topic_job_overrides(settings: Settings, topic: str, video: FoundVideo) -> Dict[str, Any]:
    """Settings for a job created from a topic: context, trend query, credit line, private lock."""
    footer = settings.youtube_description_footer.strip()
    credit = video.credit()
    if credit and credit not in footer:
        footer = (footer + "\n" + credit).strip()
    return {
        "content_context": f"{topic} — {settings.content_context}" if settings.content_context else topic,
        "trend_query": settings.trend_query or topic,
        "youtube_description_footer": footer,
        "private_only": False,
    }
