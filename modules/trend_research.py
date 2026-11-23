"""Trending-keyword research with the YouTube Data API v3 (API key, read-only).

For a topic (e.g. "GTA 6") it looks at what is performing *right now*:

* ``search.list``  — recent Shorts about the topic, ordered by views (100 quota units)
* ``videos.list``  — tags / titles / view counts of those videos           (1 unit)
* ``videos.list chart=mostPopular`` — what is trending in the category      (1 unit)

Keywords are weighted by log(views) and frequency, so terms that recur in
high-performing recent videos rank first. Results are cached per topic per
day, so re-running a job costs no quota. The LLM only receives these as
*candidates* — it keeps the ones that genuinely describe the clip (irrelevant
tags are spam under YouTube policy and hurt reach rather than help).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import Settings
from core.utils import atomic_write_json, read_json, slugify

API = "https://www.googleapis.com/youtube/v3"
STOPWORDS = set("""
a an the and or but if of to in on at for with from by is are was were be been it its this that these those
i me my we our you your he she they them his her their what which who how why when where not no so too very
just than then there here out up down over under again new video videos shorts short youtube official full
part ep episode vs live stream streaming gameplay gaming game games clip clips best top watch play plays
playing first day today now get got make made one two all more most can will do does did amp
""".split())
_WORD_RE = re.compile(r"[#@]?[\w][\w'’\-]*", re.UNICODE)


class TrendError(RuntimeError):
    pass


def derive_query(title: str, context: str = "") -> str:
    """Pick a short search phrase from a video title like 'xQc Plays GTA 6 Early | GTA 6 Playthrough #1'."""
    parts = [p.strip() for p in re.split(r"[|\-–—:•\[\]()]", title or "") if p.strip()]
    candidates = parts or [context or ""]
    # prefer the shortest meaningful part that has a capitalised token or a digit (game names, titles)
    scored = sorted(candidates, key=lambda p: (not re.search(r"[A-Z0-9]", p), len(p)))
    q = re.sub(r"#\d+|\b(playthrough|gameplay|part|episode|ep)\b.*$", "", scored[0], flags=re.I).strip()
    if len(q) < 3:
        q = (context or title or "").strip()
    return q[:60]


class TrendResearcher:
    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self.s = settings
        self.log = logger or logging.getLogger("shorts.trends")
        self.cache_dir = settings.highlights_dir / "_trends"

    def available(self) -> bool:
        return bool(self.s.youtube_api_key)

    # ------------------------------------------------------------------ http
    def _get(self, endpoint: str, **params: Any) -> Dict[str, Any]:
        params["key"] = self.s.youtube_api_key
        try:
            r = httpx.get(f"{API}/{endpoint}", params=params, timeout=30)
        except httpx.HTTPError as exc:
            raise TrendError(f"YouTube API request failed: {exc}") from exc
        if r.status_code != 200:
            try:
                err = r.json()["error"]
                reason = (err.get("errors") or [{}])[0].get("reason", "")
                msg = f"{err.get('message', r.text)} ({reason})"
            except Exception:
                msg = r.text[:300]
            raise TrendError(f"YouTube API {endpoint} HTTP {r.status_code}: {msg}")
        return r.json()

    def _video_details(self, ids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(0, len(ids), 50):
            data = self._get("videos", part="snippet,statistics", id=",".join(ids[i:i + 50]), maxResults=50)
            out.extend(data.get("items", []))
        return out

    # ------------------------------------------------------------------ public
    def check_key(self) -> str:
        data = self._get("videoCategories", part="snippet", regionCode=self.s.youtube_region)
        return f"API key OK ({len(data.get('items', []))} categories in {self.s.youtube_region})"

    def research(self, query: str, force: bool = False) -> Dict[str, Any]:
        """Return {"query", "keywords": [{term, score}], "hashtags": [...], "top_titles": [...]}."""
        if not self.available():
            raise TrendError("YOUTUBE_API_KEY is not set")
        query = (query or "").strip()
        if not query:
            raise TrendError("empty trend query")
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cache = self.cache_dir / f"{slugify(query, 50)}_{self.s.youtube_region}_{today}.json"
        cached = read_json(cache)
        if cached and not force:
            return cached

        after = (datetime.now(timezone.utc) - timedelta(days=self.s.trend_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        search = self._get(
            "search", part="snippet", q=query, type="video", order="viewCount", videoDuration="short",
            publishedAfter=after, regionCode=self.s.youtube_region,
            relevanceLanguage=self.s.youtube_language or "en", maxResults=50, safeSearch="none",
        )
        ids = [it["id"]["videoId"] for it in search.get("items", []) if it.get("id", {}).get("videoId")]
        videos = self._video_details(ids) if ids else []

        popular: List[Dict[str, Any]] = []
        try:
            popular = self._get("videos", part="snippet,statistics", chart="mostPopular", maxResults=50,
                                regionCode=self.s.youtube_region,
                                videoCategoryId=self.s.youtube_category_id).get("items", [])
        except TrendError as exc:  # some region/category combinations have no chart
            self.log.debug("mostPopular chart unavailable: %s", exc)

        result = self._aggregate(query, videos, popular)
        atomic_write_json(cache, result)
        self.log.info("Trend research '%s': %d recent videos, top keywords: %s", query, len(videos),
                      ", ".join(k["term"] for k in result["keywords"][:10]))
        return result

    # ------------------------------------------------------------------ scoring
    def _aggregate(self, query: str, videos: List[Dict[str, Any]], popular: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_tokens = {t.lower() for t in re.findall(r"\w+", query)}
        scores: Dict[str, float] = defaultdict(float)
        display: Dict[str, str] = {}
        hashtag_scores: Counter = Counter()

        def add(term: str, weight: float) -> None:
            t = re.sub(r"\s+", " ", term.strip().strip("#'\"’")).strip()
            key = t.lower()
            if len(key) < 2 or len(key) > 40 or key in STOPWORDS or key.isdigit():
                return
            scores[key] += weight
            display.setdefault(key, t)

        def harvest(items: List[Dict[str, Any]], factor: float) -> None:
            for v in items:
                sn = v.get("snippet", {})
                views = int(v.get("statistics", {}).get("viewCount", 0) or 0)
                w = factor * (1.0 + math.log10(1 + views))
                for tag in sn.get("tags", []) or []:
                    add(tag, w)
                text = f"{sn.get('title', '')} {sn.get('description', '')[:300]}"
                for tok in _WORD_RE.findall(text):
                    if tok.startswith("#"):
                        h = tok[1:]
                        if 2 <= len(h) <= 30 and h.lower() != "shorts":
                            hashtag_scores[h] += w
                            add(h, w * 0.5)
                    elif not tok.startswith("@"):
                        add(tok, w * 0.3)
                # two-word phrases from titles ("vice city", "gta 6")
                words = [t.lower() for t in re.findall(r"\w+", sn.get("title", ""))]
                for a, b in zip(words, words[1:]):
                    if a not in STOPWORDS or b not in STOPWORDS:
                        add(f"{a} {b}", w * 0.4)

        harvest(videos, 1.0)
        # trending-chart terms only count when they overlap the topic, to stay relevant
        topical = [v for v in popular
                   if q_tokens & {t.lower() for t in re.findall(r"\w+", v.get("snippet", {}).get("title", "") + " "
                                                             + " ".join(v.get("snippet", {}).get("tags", []) or []))}]
        harvest(topical, 0.7)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        keywords = [{"term": display[k], "score": round(s, 2)} for k, s in ranked[:60]]
        hashtags = ["#" + re.sub(r"[^\w]", "", h) for h, _ in hashtag_scores.most_common(20)]
        top = sorted(videos, key=lambda v: int(v.get("statistics", {}).get("viewCount", 0) or 0), reverse=True)
        return {
            "query": query,
            "region": self.s.youtube_region,
            "researched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "videos_analysed": len(videos),
            "keywords": keywords,
            "hashtags": [h for h in hashtags if len(h) > 2],
            "top_titles": [{"title": v["snippet"]["title"],
                            "views": int(v.get("statistics", {}).get("viewCount", 0) or 0)} for v in top[:15]],
        }


def format_for_prompt(trends: Optional[Dict[str, Any]], max_terms: int = 35) -> str:
    if not trends or not trends.get("keywords"):
        return "(no trend data available — rely on the transcript)"
    lines = [f"Search topic: {trends['query']} (region {trends.get('region', '')}, "
             f"{trends.get('videos_analysed', 0)} recent Shorts analysed)",
             "Trending keywords (most weight first): " + ", ".join(k["term"] for k in trends["keywords"][:max_terms])]
    if trends.get("hashtags"):
        lines.append("Trending hashtags: " + " ".join(trends["hashtags"][:15]))
    if trends.get("top_titles"):
        lines.append("Titles of the best-performing recent Shorts (for style reference only, never copy):")
        lines += [f"- {t['title']} ({t['views']:,} views)" for t in trends["top_titles"][:8]]
    return "\n".join(lines)


def load_source_tags(downloads_dir: Path, video_id: str) -> List[str]:
    info = read_json(downloads_dir / video_id / "info.json") or {}
    return [str(t) for t in (info.get("tags") or [])][:30]
