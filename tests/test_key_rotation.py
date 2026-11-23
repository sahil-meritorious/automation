"""Groq key rotation on rate limits (no network)."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from core.key_pool import KeyPool, NoValidKeyError  # noqa: E402


def get_settings(**kw):
    """Isolated settings: never read the real .env (it may hold real keys)."""
    s = Settings(_env_file=None, **kw)
    s.ensure_dirs()
    return s


class TestKeyPool(unittest.TestCase):
    def test_switch_and_wait(self):
        pool = KeyPool("t", ["a", "b", ""])
        self.assertEqual(pool.pick(), ("a", 0.0))
        self.assertTrue(pool.cooldown("a", 30))
        self.assertEqual(pool.pick()[0], "b")
        self.assertFalse(pool.cooldown("b", 10))  # nothing free now
        key, wait = pool.pick()
        self.assertEqual(key, "b")  # shortest cooldown
        self.assertTrue(5 < wait <= 10)

    def test_invalid_keys(self):
        pool = KeyPool("t", ["a", "b"])
        self.assertTrue(pool.invalid("a", "401"))
        self.assertEqual(pool.pick()[0], "b")
        self.assertFalse(pool.invalid("b", "401"))
        with self.assertRaises(NoValidKeyError):
            pool.pick()


def _rate_limit_error(msg="Rate limit reached. Please try again in 7.5s"):
    import openai

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "8"})
    return openai.RateLimitError(msg, response=resp, body=None)


class TestLLMRotation(unittest.TestCase):
    def test_rate_limit_switches_key_immediately(self):
        from core.llm_client import LLMClient

        s = get_settings(data_dir=Path(tempfile.mkdtemp()), llm_provider="groq", groq_api_key="k-primary-x1",
                         groq_api_key_1="k-second-x1", llm_tokens_per_minute=-1, llm_min_interval=0)
        llm = LLMClient(s)
        used = []

        def fake_complete(messages, client=None):
            used.append(client.api_key)
            if client.api_key == "k-primary-x1":
                raise _rate_limit_error()
            return '{"ok": true}'

        t0 = time.time()
        with mock.patch.object(llm, "_complete", side_effect=fake_complete):
            self.assertEqual(llm.chat_json("sys", "user", what="t"), {"ok": True})
        self.assertEqual(used, ["k-primary-x1", "k-second-x1"])
        self.assertLess(time.time() - t0, 2, "should not wait when another key is free")


class TestWhisperRotation(unittest.TestCase):
    def test_429_switches_key(self):
        from modules.audio_chunker import AudioChunk
        from modules.transcription import GroqTranscriber

        tmp = Path(tempfile.mkdtemp())
        audio = tmp / "c.flac"
        audio.write_bytes(b"x" * 100)
        s = get_settings(data_dir=tmp, groq_api_key="w-key-a-1", groq_api_key_2="w-key-b-1")
        tr = GroqTranscriber(s)
        seen = []

        def fake_post(url, headers=None, **kw):
            key = headers["Authorization"].split()[1]
            seen.append(key)
            req = httpx.Request("POST", url)
            if key == "w-key-a-1":
                return httpx.Response(429, request=req, headers={"retry-after": "60"}, text="rate limited")
            return httpx.Response(200, request=req, json={"segments": [], "words": [], "text": ""})

        chunk = AudioChunk(index=0, start=0, end=30, own_start=0, own_end=30, path=str(audio), size_bytes=100)
        with mock.patch("modules.transcription.httpx.post", side_effect=fake_post):
            t0 = time.time()
            res = tr._request(chunk)
        self.assertEqual(seen, ["w-key-a-1", "w-key-b-1"])
        self.assertIn("segments", res)
        self.assertLess(time.time() - t0, 3)


if __name__ == "__main__":
    unittest.main()
