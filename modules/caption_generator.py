"""Animated ASS subtitle generation (rendered by FFmpeg/libass).

Styles
  dynamic  – bold UPPERCASE 1–3 word groups, the spoken word pops in yellow,
             keywords in green, each group pops in.
  minimal  – clean sentence-case phrases on a soft translucent box.
  karaoke  – phrase-level karaoke: words fill with colour as they are spoken.

Coordinates use a fixed 1080x1920 canvas (PlayRes); libass scales to the
actual output resolution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from core.models import Word
from core.utils import strip_unsupported_chars

PLAY_W, PLAY_H = 1080, 1920

# ASS colours are &HAABBGGRR
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
YELLOW = "&H0000E6FF"
GREEN = "&H0047FF6E"
SOFT_BOX = "&H8C000000"

STRONG_WORDS = {
    "what", "wow", "insane", "crazy", "no", "yes", "never", "always", "money", "dead", "killed", "huge",
    "unbelievable", "impossible", "omg", "bro", "run", "help", "wait", "secret", "hidden", "million",
    "police", "cops", "crash", "boom", "fire", "best", "worst", "first", "last", "why", "how",
}


@dataclass
class StyleSpec:
    font_size: int
    outline: float
    shadow: float
    border_style: int
    primary: str
    secondary: str
    outline_colour: str
    back_colour: str
    bold: int
    max_words: int
    max_chars: int
    uppercase: bool


STYLES = {
    "dynamic": StyleSpec(84, 7, 3, 1, WHITE, WHITE, BLACK, "&H96000000", -1, 3, 20, True),
    "minimal": StyleSpec(64, 16, 0, 3, WHITE, WHITE, SOFT_BOX, SOFT_BOX, -1, 7, 34, False),
    "karaoke": StyleSpec(78, 6, 2, 1, YELLOW, WHITE, BLACK, "&H96000000", -1, 6, 30, True),
}


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _display(word: str, uppercase: bool) -> str:
    w = strip_unsupported_chars(word.strip())
    if uppercase:
        w = re.sub(r"[,.;:…\"“”]+$", "", w)  # keep ! and ?
        w = re.sub(r"^[\"“”]+", "", w)
        return w.upper()
    return w


def group_words(words: Sequence[Word], max_words: int, max_chars: int, max_gap: float = 0.7) -> List[List[Word]]:
    groups: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if cur:
            chars = sum(len(x.word) + 1 for x in cur) + len(w.word)
            if len(cur) >= max_words or chars > max_chars or w.start - cur[-1].end > max_gap:
                groups.append(cur)
                cur = []
        cur.append(w)
        if w.word.rstrip().endswith((".", "!", "?")) and len(cur) >= 1:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


class CaptionGenerator:
    def __init__(
        self,
        style: str = "dynamic",
        font: str = "Arial Black",
        font_size: int = 0,
        caption_y: Optional[float] = None,
    ) -> None:
        self.style_name = style if style in STYLES else "dynamic"
        self.spec = STYLES[self.style_name]
        self.font = font or "Arial"
        self.font_size = font_size or self.spec.font_size
        self.caption_y = int(caption_y if caption_y else PLAY_H * 0.72)

    # ---------------------------------------------------------------- header
    def _header(self, with_title: bool, with_footer: bool = False) -> str:
        sp = self.spec
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {PLAY_W}",
            f"PlayResY: {PLAY_H}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Caption,{self.font},{self.font_size},{sp.primary},{sp.secondary},{sp.outline_colour},"
            f"{sp.back_colour},{sp.bold},0,0,0,100,100,1,0,{sp.border_style},{sp.outline},{sp.shadow},5,90,90,0,1",
        ]
        if with_title:
            lines.append(
                f"Style: Title,{self.font},68,{WHITE},{WHITE},{BLACK},&H96000000,-1,0,0,0,100,100,0,0,1,6,2,5,80,80,0,1"
            )
        if with_footer:
            lines.append(
                f"Style: Footer,{self.font},54,{WHITE},{WHITE},{BLACK},&H8C000000,-1,0,0,0,100,100,0,0,3,10,0,5,80,80,0,1"
            )
        lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
        return "\n".join(lines)

    @staticmethod
    def _event(start: float, end: float, text: str, style: str = "Caption", layer: int = 0) -> str:
        return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"

    # ---------------------------------------------------------------- body
    def _is_key(self, w: Word, keywords: set) -> bool:
        n = _norm(w.word)
        return bool(n) and (n in keywords or (self.style_name == "dynamic" and n in STRONG_WORDS))

    def _dynamic(self, groups: List[List[Word]], keywords: set, end_limit: float) -> List[str]:
        events = []
        pos = f"\\an5\\pos({PLAY_W // 2},{self.caption_y})"
        for gi, g in enumerate(groups):
            g_end = g[-1].end + 0.25
            if gi + 1 < len(groups):
                g_end = min(g_end, groups[gi + 1][0].start)
            g_end = min(max(g_end, g[-1].end), end_limit)
            for k, w in enumerate(g):
                st = g[0].start if k == 0 else w.start
                en = g[k + 1].start if k + 1 < len(g) else g_end
                if en - st < 0.04:
                    continue
                parts = []
                for m, x in enumerate(g):
                    txt = ass_escape(_display(x.word, True))
                    if not txt:
                        continue
                    if m == k:
                        parts.append(f"{{\\c{YELLOW}&\\fscx112\\fscy112}}{txt}{{\\r}}")
                    elif self._is_key(x, keywords):
                        parts.append(f"{{\\c{GREEN}&}}{txt}{{\\r}}")
                    else:
                        parts.append(txt)
                if not parts:
                    continue
                pop = "\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)" if k == 0 else ""
                events.append(self._event(st, en, f"{{{pos}{pop}}}" + " ".join(parts)))
        return events

    def _minimal(self, groups: List[List[Word]], keywords: set, end_limit: float) -> List[str]:
        events = []
        pos = f"\\an5\\pos({PLAY_W // 2},{self.caption_y})"
        for gi, g in enumerate(groups):
            st = g[0].start
            en = g[-1].end + 0.3
            if gi + 1 < len(groups):
                en = min(en, groups[gi + 1][0].start)
            en = min(max(en, g[-1].end), end_limit)
            words = []
            for x in g:
                txt = ass_escape(_display(x.word, False))
                if txt and self._is_key(x, keywords):
                    txt = f"{{\\c&H0099F5FF&}}{txt}{{\\r}}"
                if txt:
                    words.append(txt)
            text = " ".join(words)
            if text:
                text = text[0].upper() + text[1:] if not text.startswith("{") else text
                events.append(self._event(st, en, f"{{{pos}\\fad(80,60)}}" + text))
        return events

    def _karaoke(self, groups: List[List[Word]], keywords: set, end_limit: float) -> List[str]:
        events = []
        pos = f"\\an5\\pos({PLAY_W // 2},{self.caption_y})"
        for gi, g in enumerate(groups):
            st = g[0].start
            en = g[-1].end + 0.35
            if gi + 1 < len(groups):
                en = min(en, groups[gi + 1][0].start)
            en = min(max(en, g[-1].end), end_limit)
            parts = []
            cursor = st
            for x in g:
                txt = ass_escape(_display(x.word, True))
                if not txt:
                    continue
                gap_cs = int(round(max(0.0, x.start - cursor) * 100))
                dur_cs = max(1, int(round((x.end - max(x.start, cursor)) * 100)))
                if gap_cs:
                    parts.append(f"{{\\k{gap_cs}}}")
                colour = f"\\1c{GREEN}&" if self._is_key(x, keywords) else ""
                parts.append(f"{{\\kf{dur_cs}{colour}}}{txt} ")
                cursor = max(cursor, x.end)
            if parts:
                events.append(self._event(st, en, f"{{{pos}}}" + "".join(parts).rstrip()))
        return events

    # ---------------------------------------------------------------- public
    def build(
        self,
        words: Iterable[Word],
        clip_duration: float,
        keywords: Sequence[str] = (),
        title: str = "",
        title_y: Optional[float] = None,
        title_duration: Optional[float] = None,
        footer_text: str = "",
        footer_y: Optional[float] = None,
    ) -> str:
        """``words`` must already be relative to the clip start (0 = first frame).
        ``footer_text`` stays on screen for the whole clip (campaign requirement such as "Link in bio")."""
        clean = [w for w in words if w.word.strip() and w.end > 0 and w.start < clip_duration]
        clean = [Word(word=w.word.strip(), start=max(0.0, w.start), end=min(clip_duration, max(w.end, w.start + 0.05)))
                 for w in clean]
        keyset = {_norm(t) for k in keywords for t in str(k).split() if _norm(t)}
        groups = group_words(clean, self.spec.max_words, self.spec.max_chars)
        end_limit = max(0.0, clip_duration - 0.05)
        builder = {"dynamic": self._dynamic, "minimal": self._minimal, "karaoke": self._karaoke}[self.style_name]
        events = builder(groups, keyset, end_limit)

        title_text = ass_escape(strip_unsupported_chars(title))
        if title_text:
            ty = int(title_y if title_y else PLAY_H * 0.16)
            tdur = min(clip_duration, title_duration) if title_duration else clip_duration
            events.insert(0, self._event(0, max(0.5, tdur - 0.05),
                                         f"{{\\an5\\pos({PLAY_W // 2},{ty})\\fad(200,250)}}{title_text}",
                                         style="Title", layer=1))
        footer = ass_escape(strip_unsupported_chars(footer_text or ""))
        if footer:
            fy = int(footer_y if footer_y else PLAY_H * 0.835)
            events.insert(0, self._event(0, max(0.5, clip_duration - 0.05),
                                         f"{{\\an5\\pos({PLAY_W // 2},{fy})}}{footer}", style="Footer", layer=1))
        return self._header(bool(title_text), bool(footer)) + "\n" + "\n".join(events) + "\n"

    def write(self, path: Path, *args, **kwargs) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(*args, **kwargs), encoding="utf-8")
        return path


def caption_spans(ass_text: str) -> List[tuple]:
    """(start, end) of every Caption event — used by the quality checker."""
    spans = []
    for line in ass_text.splitlines():
        if line.startswith("Dialogue:") and ",Caption," in line:
            parts = line.split(",", 9)
            spans.append((_parse_ass_time(parts[1]), _parse_ass_time(parts[2])))
    return spans


def _parse_ass_time(t: str) -> float:
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def srt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def build_srt(words: Sequence[Word], clip_duration: float, max_words: int = 7, max_chars: int = 42) -> str:
    """Plain subtitle track (clip-relative times) for uploading to YouTube as a caption file."""
    words = [w for w in words if w.end > 0 and w.start < clip_duration and w.word.strip()]
    blocks = []
    for i, g in enumerate(group_words(words, max_words, max_chars, max_gap=0.9), 1):
        start = max(0.0, g[0].start)
        end = min(clip_duration, max(g[-1].end, start + 0.6))
        text = " ".join(w.word.strip() for w in g)
        blocks.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(blocks)
