"""Topic → source video discovery (YouTube API mocked)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from modules.video_finder import FinderError, VideoFinder, iso_duration, topic_job_overrides  # noqa: E402


def fake_get(endpoint, **params):
    if endpoint == "channels":
        return {"items": [{"id": "UC" + "h" * 22}]} if params["forHandle"] == "@Known" else {"items": []}
    if endpoint == "search":
        return {"items": [{"id": {"videoId": v}} for v in ("a", "b", "c", "d")]}
    items = {
        "a": ("GTA 5 vs GTA 6 map comparison", "MapLab", "PT18M", "creativeCommon", "900000"),
        "b": ("Asmongold reacts to GTA 6", "ClipDump", "PT25M", "creativeCommon", "5000000"),  # re-upload
        "c": ("GTA 6 trailer 60s", "Short", "PT1M", "creativeCommon", "100"),  # too short
        "d": ("GTA 6 easter eggs explained", "Owner", "PT30M", "youtube", "1000"),  # not CC
    }
    return {"items": [{"id": i, "snippet": {"title": t, "channelTitle": ch, "channelId": "x",
                                           "publishedAt": "2026-11-01T00:00:00Z", "description": ""},
                       "contentDetails": {"duration": d}, "statistics": {"viewCount": vc}, "status": {"license": lic}}
                      for i, (t, ch, d, lic, vc) in items.items() if i in params["id"].split(",")]}


class TestFinder(unittest.TestCase):
    def setUp(self):
        self.s = get_settings(youtube_api_key="k", data_dir=Path(tempfile.mkdtemp()), topic_source="creative_commons")

    def test_filters_and_picks(self):
        f = VideoFinder(self.s)
        with mock.patch.object(f.api, "_get", side_effect=fake_get):
            found = f.find("GTA 5 vs GTA 6", count=2)
        self.assertEqual([v.video_id for v in found], ["a"])  # re-upload, too short and non-CC dropped
        o = topic_job_overrides(self.s, "GTA 5 vs GTA 6", found[0])
        self.assertIn("CC BY", o["youtube_description_footer"])
        self.assertEqual(o["trend_query"], "GTA 5 vs GTA 6")

    def test_exclude_already_processed(self):
        f = VideoFinder(self.s)
        with mock.patch.object(f.api, "_get", side_effect=fake_get), self.assertRaises(FinderError):
            f.find("GTA 5 vs GTA 6", exclude_ids=["a"])

    def test_any_public_marks_private_only(self):
        s = get_settings(youtube_api_key="k", data_dir=Path(tempfile.mkdtemp()), topic_source="any_public")
        f = VideoFinder(s)
        with mock.patch.object(f.api, "_get", side_effect=fake_get):
            found = f.find("GTA 6", count=3)
        self.assertIn("d", [v.video_id for v in found])  # non-CC allowed in testing mode
        self.assertTrue(topic_job_overrides(s, "GTA 6", found[0])["private_only"])
        self.assertFalse(topic_job_overrides(self.s, "GTA 6", found[0])["private_only"])

    def test_allowed_channels_resolve_handles(self):
        s = get_settings(youtube_api_key="k", data_dir=Path(tempfile.mkdtemp()),
                         allowed_channel_ids="UC" + "a" * 22 + ", @Known, https://youtube.com/@Missing")
        f = VideoFinder(s)
        with mock.patch.object(f.api, "_get", side_effect=fake_get):
            self.assertEqual(f.allowed_channels(), ["UC" + "a" * 22, "UC" + "h" * 22])

    def test_iso_duration(self):
        self.assertEqual(iso_duration("PT1H2M3S"), 3723)
        self.assertEqual(iso_duration("PT45S"), 45)


if __name__ == "__main__":
    unittest.main()
