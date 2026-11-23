"""Campaign briefs: parsing fallback, settings mapping, and a full offline pipeline run
(exact segments + extra AI clips + logo + on-screen text + required metadata).
Run: python -m unittest tests.test_campaign
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from core.utils import run_ffmpeg  # noqa: E402
from modules.campaign_brief import BriefParser, CampaignBrief, campaign_overrides, regex_extract  # noqa: E402
from modules.metadata_generator import enforce_brief  # noqa: E402
from tests.test_pipeline_offline import fake_transcribe  # noqa: E402

BRIEF_TEXT = """Brand X campaign. Video: https://www.youtube.com/watch?v=abcdefghijk
Use 0:10 - 0:30 and 0:50-1:10. Tag @brandx, hashtags #BrandX #ad. Clips 15-40s."""


class FakeLLM:
    def __init__(self, settings, logger=None):
        self.label = "fake:model"

    def chat_json(self, system, user, validate=None, attempts=3, what="", cancel_event=None):
        if what == "Segment direction":
            data = {"clips": [{"id": "g1", "hook_start": 14, "title": "Seg one", "hook_text": "Watch this",
                               "caption_style": "minimal", "layout": "center_crop", "punch_in": False, "zoom": False,
                               "score": 85, "clip_start": 0, "clip_end": 999},  # must be ignored
                              {"id": "g2", "hook_start": 55, "title": "Seg two", "hook_text": "No way",
                               "score": 80}]}
        elif what.startswith("Highlight detection"):
            data = {"clips": [
                {"clip_start": 12, "hook_start": 15, "clip_end": 32, "score": 90, "title": "dup of g1"},
                {"clip_start": 74, "hook_start": 78, "clip_end": 96, "score": 88, "title": "Extra moment",
                 "hook_text": "Extra!", "category": "funny"},
            ]}
        elif what == "Global re-rank":
            import re
            ids = re.findall(r"^id: (\S+)", user, flags=re.M)
            data = {"ranking": [{"id": i, "score": 85, "keep": True} for i in ids]}
        else:
            import re
            ids = re.findall(r"^id: (\S+)", user, flags=re.M)
            data = {"shorts": [{"id": i, "title": f"Title {i}", "description": "Great clip.",
                                "hashtags": ["#Shorts", "#Gaming"], "tags": ["gaming"]} for i in ids]}
        return validate(data) if validate else data


class TestBriefParsing(unittest.TestCase):
    def test_regex_fallback(self):
        s = get_settings(data_dir=Path(tempfile.mkdtemp()))
        b = BriefParser(s, None).parse(BRIEF_TEXT)
        self.assertEqual(b.source_urls, ["https://www.youtube.com/watch?v=abcdefghijk"])
        self.assertEqual([(g.start, g.end) for g in b.segments], [(10, 30), (50, 70)])
        self.assertEqual(b.required_mentions, ["@brandx"])
        self.assertIn("#BrandX", b.required_hashtags)

    def test_overrides_and_enforcement(self):
        s = get_settings(data_dir=Path(tempfile.mkdtemp()))
        b = CampaignBrief(num_clips=4, min_duration=20, max_duration=45, required_hashtags=["#BrandX", "#ad"],
                          required_mentions=["@brandx"], required_title_text="BrandX",
                          required_description_text="Code CLIP10", music_allowed=False, captions_required=True)
        o = campaign_overrides(s, b, "/tmp/logo.png", "brandx-1")
        self.assertEqual((o["num_shorts"], o["min_clip_duration"], o["max_clip_duration"]), (4, 20, 45))
        self.assertEqual(o["background_music"], "")
        self.assertTrue(o["captions_enabled"])
        get_settings(**o)  # must validate
        r = enforce_brief("Crazy boat moment", "Look at this.", ["#Shorts", "#GTA6"], b)
        self.assertIn("BrandX", r["title"])
        self.assertIn("Code CLIP10", r["description"])
        self.assertIn("@brandx", r["description"])
        self.assertEqual(r["hashtags"][:3], ["#Shorts", "#BrandX", "#ad"])

    def test_regex_mentions_ignore_emails(self):
        self.assertEqual(regex_extract("mail a@b.com, tag @brand.")["required_mentions"], ["@brand"])


class TestCampaignPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="campaign_e2e_"))
        cls.video = cls.tmp / "brand_video.mp4"
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=100",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=100",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(cls.video)])
        cls.logo = cls.tmp / "logo.png"
        run_ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=200x80", "-frames:v", "1", str(cls.logo)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_segments_plus_extra(self):
        from core.pipeline import ShortsPipeline

        brief = CampaignBrief(name="Brand X", segments=[{"start": 10, "end": 30}, {"start": 50, "end": 70}],
                              extra_clips=1, required_hashtags=["#BrandX"], required_mentions=["@brandx"],
                              required_description_text="Code CLIP10", on_screen_text="Link in bio",
                              logo_required=True, logo_position="top_left")
        base = get_settings(data_dir=self.tmp / "data", groq_api_key="test", audio_chunk_seconds=40,
                            audio_chunk_overlap=5, analysis_window_seconds=120, min_clip_duration=15,
                            max_clip_duration=40, output_width=360, output_height=640, video_preset="ultrafast",
                            min_free_disk_gb=0.05, layout="auto", caption_style="auto")
        settings = get_settings(**{**base.model_dump(), **campaign_overrides(base, brief, str(self.logo), "brand-x")})
        with mock.patch("modules.transcription.GroqTranscriber._request", fake_transcribe), \
                mock.patch("core.pipeline.LLMClient", FakeLLM):
            res = ShortsPipeline(settings).run(str(self.video))
        self.assertEqual(res["status"], "completed", res.get("error"))
        shorts = res["shorts"]
        self.assertEqual(len(shorts), 3)
        self.assertEqual([(m["clip_start"], m["clip_end"]) for m in shorts[:2]], [(10, 30), (50, 70)])
        self.assertEqual(shorts[0]["hook_start"], 14)
        self.assertEqual(shorts[0]["layout"], "center_crop")
        self.assertTrue(70 <= shorts[2]["clip_start"] <= 74)  # extra AI clip (snapped to a sentence); overlap dropped
        for m in shorts:
            self.assertEqual(m["status"], "exported", m.get("error"))
            self.assertIn("#BrandX", m["hashtags"])
            self.assertIn("Code CLIP10", m["description"])
            self.assertIn("@brandx", m["description"])
            self.assertTrue(any("Logo" in r for r in m["requirements"]))
        self.assertIn("brand-x", res["output_dir"])
        report = json.loads((Path(res["output_dir"]) / "report.json").read_text())
        self.assertTrue(any("segment given by the campaign" in c["reject_reason"] for c in report["candidates"]))


if __name__ == "__main__":
    unittest.main()
