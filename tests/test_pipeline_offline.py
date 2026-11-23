"""End-to-end pipeline test with Groq and the LLM mocked out.

Generates a synthetic 100 s video, then runs the real pipeline: chunking,
(mocked) transcription, merge, (mocked) highlight detection, ranking,
metadata, real FFmpeg rendering, QC, reports, DB rows and resume/caching.
Run: python -m unittest tests.test_pipeline_offline
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from core.utils import run_ffmpeg  # noqa: E402


def fake_transcribe(self, chunk):
    """Pretend speech: a word every 0.5 s named by its absolute half-second."""
    words, t = [], 0.0
    while chunk.start + t + 0.3 <= chunk.end:
        n = int(round((chunk.start + t) * 2))
        words.append({"word": f"word{n}" + ("." if n % 12 == 11 else ""), "start": t, "end": t + 0.3})
        t += 0.5
    segs = []
    for k in range(0, len(words), 12):
        g = words[k:k + 12]
        segs.append({"start": g[0]["start"], "end": g[-1]["end"], "text": " ".join(w["word"] for w in g),
                     "no_speech_prob": 0.01, "avg_logprob": -0.1})
    return {"segments": segs, "words": words, "language": "english"}


class FakeLLM:
    calls = {"detect": 0, "rerank": 0, "meta": 0}

    def __init__(self, settings, logger=None):
        self.label = "fake:model"

    def chat_json(self, system, user, validate=None, attempts=3, what="", cancel_event=None):
        if what.startswith("Highlight detection"):
            FakeLLM.calls["detect"] += 1
            m = re.search(r"covering (\d+)s to (\d+)s", user)
            start = float(m.group(1))
            # invalid JSON shape first → validator must reject, real client would retry
            data = {"clips": [
                {"clip_start": start + 5, "hook_start": start + 9, "clip_end": start + 30, "score": 88,
                 "subscores": {"hook": 90, "entertainment": 85}, "category": "funny",
                 "reason": "setup then payoff", "title": f"Moment at {start:.0f}", "keywords": ["word20"],
                 "hook_text": "Wait for it", "layout": "center_crop", "caption_style": "karaoke",
                 "punch_in": False, "zoom": True, "virality": "rewatch loop"},
                {"clip_start": start + 6, "hook_start": start + 10, "clip_end": start + 31, "score": 70,
                 "category": "funny", "title": "duplicate"},
            ]}
        elif what == "Global re-rank":
            FakeLLM.calls["rerank"] += 1
            ids = re.findall(r"^id: (\S+)", user, flags=re.M)
            data = {"ranking": [{"id": i, "score": 80, "keep": True, "note": "ok"} for i in ids]}
        else:
            FakeLLM.calls["meta"] += 1
            ids = re.findall(r"^id: (\S+)", user, flags=re.M)
            data = {"shorts": [{"id": i, "title": f"Great title {i}", "description": "desc",
                                "hashtags": ["#Shorts", "#Test"]} for i in ids]}
        return validate(data) if validate else data


class TestPipelineOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shorts_e2e_"))
        cls.video = cls.tmp / "my_gameplay.mp4"
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=100",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=100",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(cls.video)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_full_pipeline(self):
        from core.pipeline import ShortsPipeline

        settings = get_settings(
            data_dir=self.tmp / "data", groq_api_key="test", audio_chunk_seconds=40, audio_chunk_overlap=5,
            analysis_window_seconds=120, analysis_overlap_seconds=10, num_shorts=3, min_clip_duration=15,
            max_clip_duration=40, output_width=360, output_height=640, video_preset="ultrafast",
            render_workers=2, layout="auto", caption_style="auto", punch_in_mode="auto", zoom_mode="auto",
            overlay_text="auto", min_free_disk_gb=0.05,
        )
        with mock.patch("modules.transcription.GroqTranscriber._request", fake_transcribe), \
                mock.patch("core.pipeline.LLMClient", FakeLLM):
            res = ShortsPipeline(settings).run(str(self.video))
            self.assertEqual(res["status"], "completed", res.get("error"))
            shorts = res["shorts"]
            self.assertGreaterEqual(len(shorts), 1)
            for m in shorts:
                self.assertEqual(m["status"], "exported", m.get("error"))
                self.assertTrue(Path(m["output_path"]).exists())
                self.assertTrue(m["qc"]["passed"])
                self.assertTrue(Path(m["output_path"]).with_suffix(".srt").exists())
                # the AI's editing decisions were applied
                self.assertEqual((m["layout"], m["caption_style"], m["punch_in"], m["zoom"]),
                                 ("center_crop", "karaoke", False, True))
                self.assertEqual(m["hook_text"], "Wait for it")
                self.assertEqual(m["decided_by"]["layout"], "ai")
            out = Path(res["output_dir"])
            for f in ("report.json", "report.md", "metadata.json", "metadata.csv"):
                self.assertTrue((out / f).exists(), f)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(any(c["reject_reason"].startswith("duplicate") for c in report["candidates"]))

            tr = json.loads((settings.transcripts_dir / res["video"]["video_id"] / "transcript.json").read_text())
            words = [w["word"].rstrip(".") for s in tr["transcript"] for w in s["words"]]
            self.assertEqual(len(words), len(set(words)), "overlap produced duplicate words")

            # Second run must reuse every checkpoint (no new LLM detection calls, no re-render)
            before = dict(FakeLLM.calls)
            res2 = ShortsPipeline(settings).run(str(self.video))
            self.assertEqual(res2["status"], "completed")
            self.assertEqual(FakeLLM.calls["detect"], before["detect"])


if __name__ == "__main__":
    unittest.main()
