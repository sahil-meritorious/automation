"""Unit tests for pure logic (no network, no FFmpeg). Run: python -m unittest discover tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from core.llm_client import parse_json_loose  # noqa: E402
from core.models import ClipCandidate, Segment, Transcript, Word  # noqa: E402
from core.rate_limiter import RateLimiter  # noqa: E402
from core.utils import parse_timestamp, slugify  # noqa: E402
from modules.audio_chunker import AudioChunk, plan_chunks  # noqa: E402
from modules.caption_generator import CaptionGenerator, caption_spans  # noqa: E402
from modules.downloader import parse_youtube_url, validate_source  # noqa: E402
from modules.highlight_detector import build_windows  # noqa: E402
from modules.highlight_ranker import HighlightRanker, is_duplicate  # noqa: E402
from modules.metadata_generator import clean_hashtags, clean_title  # noqa: E402
from modules.transcript_merger import merge_chunk_results  # noqa: E402


def make_transcript(duration: float = 300.0, step: float = 0.5) -> Transcript:
    segs, t, i = [], 0.0, 0
    while t + 5 <= duration:
        words = []
        for k in range(10):
            text = f"w{i}" + ("." if k == 9 else "")
            words.append(Word(word=text, start=t + k * step, end=t + k * step + 0.4))
            i += 1
        segs.append(Segment(start=t, end=t + 10 * step, text=" ".join(w.word for w in words), words=words))
        t += 10 * step
    return Transcript(video_id="x", duration=duration, transcript=segs)


class TestUtils(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(parse_timestamp("12:20"), 740)
        self.assertEqual(parse_timestamp("1:00:01.5"), 3601.5)
        self.assertEqual(parse_timestamp(12), 12.0)
        self.assertEqual(parse_timestamp("75.5s"), 75.5)

    def test_slugify(self):
        self.assertEqual(slugify("You Won't Believe THIS!! 😱"), "you-won-t-believe-this")
        self.assertEqual(slugify("😱"), "short")

    def test_youtube_urls(self):
        for url in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=3",
                    "https://www.youtube.com/shorts/dQw4w9WgXcQ", "https://youtube.com/watch?feature=x&v=dQw4w9WgXcQ"):
            self.assertEqual(parse_youtube_url(url), "dQw4w9WgXcQ")
        self.assertFalse(validate_source("https://www.youtube.com/playlist?list=PL123")[0])
        self.assertFalse(validate_source("")[0])


class TestJsonParsing(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(parse_json_loose('```json\n{"clips": []}\n```'), {"clips": []})

    def test_prose(self):
        self.assertEqual(parse_json_loose('Sure! Here you go: {"a": 1} hope it helps'), {"a": 1})

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_json_loose("no json here")


class TestModels(unittest.TestCase):
    def test_candidate_normalisation(self):
        c = ClipCandidate.model_validate({"clip_start": "12:20", "hook_start": "12:35", "clip_end": "12:55",
                                          "score": 9.1, "subscores": {"hook": 8, "clarity": 7}, "keywords": "a, b"})
        self.assertEqual((c.clip_start, c.hook, c.clip_end), (740, 755, 775))
        self.assertAlmostEqual(c.score, 91)
        self.assertEqual(c.subscores["hook"], 80)
        self.assertEqual(c.keywords, ["a", "b"])

    def test_candidate_hook_clamped(self):
        c = ClipCandidate(clip_start=10, clip_end=40, hook_start=100)
        self.assertEqual(c.hook, 40)

    def test_candidate_invalid(self):
        with self.assertRaises(Exception):
            ClipCandidate(clip_start=10, clip_end=10.5)


class TestChunking(unittest.TestCase):
    def test_plan_covers_timeline(self):
        chunks = plan_chunks(3700, 600, 5)
        self.assertEqual(chunks[0].start, 0)
        self.assertAlmostEqual(chunks[-1].end, 3700)
        for a, b in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(a.own_end, b.own_start)
            self.assertLess(b.start, a.end)  # overlap exists

    def test_short_audio_single_chunk(self):
        self.assertEqual(len(plan_chunks(100, 600, 5)), 1)


class TestMerger(unittest.TestCase):
    def test_overlap_dedup_and_offsets(self):
        duration = 95.0
        chunks = plan_chunks(duration, 30, 6)
        # Every 0.5 s a word "wN" on the absolute timeline; each chunk "hears" its range.
        results = []
        for c in chunks:
            words, segs = [], []
            t = 0.0
            while c.start + t + 0.3 <= c.end:
                n = int(round((c.start + t) * 2))
                words.append({"word": f"w{n}", "start": t, "end": t + 0.3})
                t += 0.5
            for k in range(0, len(words), 8):
                grp = words[k:k + 8]
                segs.append({"start": grp[0]["start"], "end": grp[-1]["end"],
                             "text": " ".join(w["word"] for w in grp), "no_speech_prob": 0.01, "avg_logprob": -0.2})
            results.append((c, {"segments": segs, "words": words, "language": "en"}))
        tr = merge_chunk_results(results, "vid", duration, "test")
        all_words = tr.all_words()
        names = [w.word for w in all_words]
        self.assertEqual(len(names), len(set(names)), "duplicate words survived the merge")
        expected = int((duration - 0.3) * 2) + 1
        self.assertGreaterEqual(len(names), expected - 2)
        for w in all_words:  # timestamps restored to the original timeline
            self.assertAlmostEqual(w.start, int(w.word[1:]) / 2, places=2)

    def test_hallucination_filtered(self):
        c = AudioChunk(0, 0, 30, 0, 31)
        raw = {"segments": [
            {"start": 0, "end": 3, "text": "Hello there.", "no_speech_prob": 0.01, "avg_logprob": -0.2},
            {"start": 20, "end": 22, "text": "Thank you.", "no_speech_prob": 0.95, "avg_logprob": -1.0}]}
        tr = merge_chunk_results([(c, raw)], "v", 30, "m")
        self.assertEqual([s.text for s in tr.transcript], ["Hello there."])
        self.assertTrue(tr.transcript[0].words)  # synthesised


class TestRanker(unittest.TestCase):
    def setUp(self):
        self.s = get_settings(min_clip_duration=15, max_clip_duration=60, num_shorts=2, min_score=60,
                              llm_rerank=False)
        self.tr = make_transcript(400)

    def test_refine_long_clip_keeps_hook(self):
        c = ClipCandidate(clip_start=10, hook_start=90, clip_end=150, score=90)
        r = HighlightRanker(self.s).refine(c, self.tr)
        self.assertIsNotNone(r)
        self.assertLessEqual(r.duration, self.s.hard_max_duration + 0.6)
        self.assertLessEqual(r.clip_start, 90)
        self.assertGreaterEqual(r.clip_end, 90)

    def test_refine_extends_short_clip(self):
        c = ClipCandidate(clip_start=100, hook_start=101, clip_end=106, score=90)
        r = HighlightRanker(self.s).refine(c, self.tr)
        self.assertGreaterEqual(r.duration, 15)

    def test_selection_dedup_and_threshold(self):
        cands = [
            ClipCandidate(id="a", clip_start=20, hook_start=25, clip_end=50, score=90),
            ClipCandidate(id="b", clip_start=22, hook_start=26, clip_end=52, score=85),  # duplicate of a
            ClipCandidate(id="c", clip_start=200, hook_start=205, clip_end=235, score=80),
            ClipCandidate(id="d", clip_start=300, hook_start=305, clip_end=335, score=40),  # too weak
        ]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            from core.models import VideoInfo
            selected, all_c = HighlightRanker(self.s).rank(cands, self.tr, VideoInfo(video_id="x", source="x"), Path(d))
        self.assertEqual([c.id for c in selected], ["a", "c"])
        self.assertTrue(is_duplicate(cands[0], cands[1]))
        self.assertEqual(len(all_c), 4)


class TestCaptions(unittest.TestCase):
    def test_all_styles_produce_events(self):
        words = [Word(word=w, start=i * 0.4, end=i * 0.4 + 0.3) for i, w in enumerate("this is so crazy bro what".split())]
        for style in ("dynamic", "minimal", "karaoke"):
            ass = CaptionGenerator(style).build(words, 5.0, keywords=["crazy"], title="Hi")
            spans = caption_spans(ass)
            self.assertTrue(spans, style)
            self.assertTrue(all(0 <= a <= b <= 5.0 for a, b in spans), style)
            self.assertIn("Style: Title", ass)


class TestMetadataHelpers(unittest.TestCase):
    def test_title_and_tags(self):
        self.assertEqual(clean_title('"THIS IS COMPLETELY INSANE!!! #gta"'), "This Is Completely Insane!")
        tags = clean_hashtags(["gta6", "#Funny Moments", "#gta6"], "funny", "GTA gameplay")
        self.assertEqual(tags[0], "#Shorts")
        self.assertIn("#gta6", tags)
        self.assertEqual(len(tags), len({t.lower() for t in tags}))


class TestWindowsAndLimiter(unittest.TestCase):
    def test_windows(self):
        w = build_windows(3600, 600, 60)
        self.assertEqual(w[0][0], 0)
        self.assertEqual(w[-1][1], 3600)
        self.assertTrue(all(b[0] < a[1] for a, b in zip(w, w[1:])))

    def test_limiter_units(self):
        rl = RateLimiter("t", None, [(3600, 100)])
        rl.acquire(60)
        self.assertGreater(rl._wait_time(__import__("time").time(), 60), 3000)


if __name__ == "__main__":
    unittest.main()
