"""YouTube upload + trend research logic (Google APIs mocked, no network).
Run: python -m unittest tests.test_youtube
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from core.models import ShortMetadata, Word  # noqa: E402
from database.database import Database  # noqa: E402
from modules.caption_generator import build_srt  # noqa: E402
from modules.metadata_generator import build_description, clean_tags, write_metadata_files  # noqa: E402
from modules.trend_research import TrendResearcher, derive_query, format_for_prompt  # noqa: E402
from modules.youtube_uploader import QuotaExceededError, YouTubeUploader, client_config  # noqa: E402


class TestMetadataHelpers(unittest.TestCase):
    def test_clean_tags_limits_and_dedupes(self):
        tags = clean_tags(["GTA 6", "gta 6", "#police chase", "<bad>", "x" * 70] + [f"tag number {i}" for i in range(80)])
        self.assertEqual(tags[0], "GTA 6")
        self.assertIn("police chase", tags)
        self.assertNotIn("gta 6", tags)
        self.assertTrue(all("<" not in t and len(t) <= 60 for t in tags))
        self.assertLessEqual(sum(len(t) + (2 if " " in t else 0) + 1 for t in tags), 450)

    def test_build_description(self):
        m = ShortMetadata(rank=1, title="t", description="Wild chase.", hashtags=["#Shorts", "#GTA6", "#Funny"],
                          filename="a.mp4", clip_start=0, hook_start=1, clip_end=20, duration=20)
        d = build_description(m, "Credit: my channel")
        self.assertEqual(d, "Wild chase.\n\nCredit: my channel\n\n#Shorts #GTA6 #Funny")

    def test_srt(self):
        words = [Word(word=w, start=i * 0.4, end=i * 0.4 + 0.3) for i, w in enumerate("hello there. how are you".split())]
        srt = build_srt(words, 5.0)
        self.assertTrue(srt.startswith("1\n00:00:00,000 --> "))
        self.assertIn("hello there.", srt)
        self.assertIn("2\n", srt)

    def test_derive_query(self):
        self.assertEqual(derive_query("xQc Plays GTA 6 Early | GTA 6 Playthrough #1"), "GTA 6")
        self.assertTrue(derive_query("", "Minecraft speedrun"))


class TestTrendAggregate(unittest.TestCase):
    def test_aggregate_weights_views(self):
        s = get_settings(youtube_api_key="k", data_dir=Path(tempfile.mkdtemp()))
        vids = [
            {"snippet": {"title": "GTA 6 police chase gone wrong #gta6", "tags": ["gta 6", "police chase"]},
             "statistics": {"viewCount": "2000000"}},
            {"snippet": {"title": "GTA 6 funny moments", "tags": ["gta 6", "funny moments"]},
             "statistics": {"viewCount": "500"}},
        ]
        r = TrendResearcher(s)._aggregate("GTA 6", vids, [])
        terms = [k["term"].lower() for k in r["keywords"]]
        self.assertEqual(terms[0], "gta 6")
        self.assertLess(terms.index("police chase"), terms.index("funny moments"))
        self.assertIn("#gta6", r["hashtags"])
        self.assertIn("GTA 6", format_for_prompt(r))

    def test_research_uses_cache(self):
        s = get_settings(youtube_api_key="k", data_dir=Path(tempfile.mkdtemp()))
        tr = TrendResearcher(s)
        calls = []

        def fake_get(endpoint, **params):
            calls.append(endpoint)
            if endpoint == "search":
                return {"items": [{"id": {"videoId": "abc"}}]}
            return {"items": [{"snippet": {"title": "GTA 6 heist", "tags": ["gta 6"]},
                               "statistics": {"viewCount": "10"}}]}

        with mock.patch.object(tr, "_get", side_effect=fake_get):
            tr.research("GTA 6")
            n = len(calls)
            tr.research("GTA 6")
        self.assertEqual(len(calls), n, "second call should hit the cache")


class TestUploader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = get_settings(data_dir=self.tmp, youtube_client_id="id", youtube_client_secret="sec",
                              youtube_upload_captions=True)
        self.db = Database(self.tmp / "t.db")
        self.job = self.db.create_job("src", {})
        out = self.tmp / "out"
        out.mkdir()
        metas = []
        for r in (1, 2, 3):
            mp4 = out / f"0{r}_x.mp4"
            mp4.write_bytes(b"fake")
            mp4.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
            m = ShortMetadata(rank=r, title=f"Title {r}", description="desc", hashtags=["#Shorts"], tags=["gta 6"],
                              filename=mp4.name, clip_start=0, hook_start=1, clip_end=20, duration=20,
                              status="exported", output_path=str(mp4))
            metas.append(m)
            self.db.save_short(self.job, "vid", m.model_dump())
        write_metadata_files(out, metas)
        self.db.update_job(self.job, output_dir=str(out))
        self.out = out

    def _uploader(self):
        up = YouTubeUploader(self.s, self.db)
        up._yt = object()  # skip real auth
        return up

    def test_client_config_from_env(self):
        cfg = client_config(self.s)
        self.assertEqual(cfg["installed"]["client_id"], "id")

    def test_body_and_schedule(self):
        up = self._uploader()
        when = datetime.now(timezone.utc) + timedelta(hours=3)
        body = up.build_body({"title": "A <b> title", "description": "d", "hashtags": ["#Shorts"], "tags": ["x"]},
                             "public", when)
        self.assertEqual(body["status"]["privacyStatus"], "private")  # required with publishAt
        self.assertTrue(body["status"]["publishAt"].endswith("Z"))
        self.assertNotIn("<", body["snippet"]["title"])
        self.assertIn("#Shorts", body["snippet"]["description"])
        times = up._schedule(3, None, 4)
        self.assertEqual(times[1] - times[0], timedelta(hours=4))

    def test_upload_job_idempotent_and_quota(self):
        up = self._uploader()
        ids = iter(["v1", "v2", "v3"])
        with mock.patch.object(up, "upload_video", side_effect=lambda *a, **k: next(ids)), \
                mock.patch.object(up, "upload_captions") as cap:
            res = up.upload_job(self.job, privacy="unlisted")
        self.assertEqual([r["youtube_video_id"] for r in res], ["v1", "v2", "v3"])
        self.assertEqual(cap.call_count, 3)
        self.assertEqual(self.db.get_short(self.job, 2)["upload_status"], "uploaded")
        self.assertTrue((self.out / "uploads.json").exists())

        with mock.patch.object(up, "upload_video") as again:
            res2 = up.upload_job(self.job)
        again.assert_not_called()
        self.assertTrue(all(r.get("skipped") for r in res2))

    def test_private_only_job_forced_private(self):
        self.db.update_job(self.job, settings_json=json.dumps({"private_only": False}))
        up = self._uploader()
        bodies = []

        def fake(path, body, progress=None):
            bodies.append(body)
            return f"v{len(bodies)}"

        with mock.patch.object(up, "upload_video", side_effect=fake), mock.patch.object(up, "upload_captions"):
            res = up.upload_job(self.job, privacy="public", interval_hours=4)
        self.assertTrue(all(b["status"]["privacyStatus"] == "private" and "publishAt" not in b["status"]
                            for b in bodies))
        self.assertTrue(all(r["privacy"] == "private" for r in res))

    def test_quota_stops_batch(self):
        up = self._uploader()
        seq = ["v1", QuotaExceededError("quotaExceeded")]

        def fake(*a, **k):
            x = seq.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

        with mock.patch.object(up, "upload_video", side_effect=fake), mock.patch.object(up, "upload_captions"):
            res = up.upload_job(self.job, privacy="private")
        self.assertEqual([r["upload_status"] for r in res], ["uploaded", "pending", "pending"])
        self.assertEqual(self.db.get_short(self.job, 3)["upload_status"], "pending")
        log = json.loads((self.out / "uploads.json").read_text())
        self.assertEqual(len(log), 3)


if __name__ == "__main__":
    unittest.main()
