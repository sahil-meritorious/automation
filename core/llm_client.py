"""Provider-agnostic LLM client built on the OpenAI-compatible Chat Completions API.

Supported providers (switch with LLM_PROVIDER in .env):
  gemini     – Google Gemini via its OpenAI-compatible endpoint (GEMINI_API_KEY)
  openai     – OpenAI (OPENAI_API_KEY)
  groq       – Groq-hosted open models (GROQ_API_KEY, same key as transcription)
  openrouter – OpenRouter (OPENROUTER_API_KEY)
  ollama     – local Ollama server, no key needed
  custom     – any OpenAI-compatible server (LLM_BASE_URL + LLM_API_KEY)
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from config import Settings
from core.key_pool import NoValidKeyError, get_key_pool
from core.utils import JobCancelled, interruptible_sleep

T = TypeVar("T")

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_field": "gemini_api_key",
        "json_mode": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "key_field": "openai_api_key",
        "json_mode": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key_field": "groq_api_key",
        "json_mode": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
        "key_field": "openrouter_api_key",
        "json_mode": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b",
        "key_field": None,
        "json_mode": True,
    },
    "custom": {"base_url": "", "model": "", "key_field": "llm_api_key", "json_mode": False},
}


def _retry_after(exc: Any, default: float) -> float:
    """Seconds to wait from a 429 response (retry-after header or 'try again in Xs'), capped at 90 s."""
    try:
        hdr = exc.response.headers.get("retry-after")
        if hdr:
            return min(90.0, float(hdr) + 1)
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if m:
        return min(90.0, int(m.group(1) or 0) * 60 + float(m.group(2)) + 1)
    return min(90.0, default)


def estimate_tokens(text: str) -> int:
    """Rough token count (≈3.5 characters per token for English + JSON)."""
    return int(len(text) / 3.5) + 1


class LLMError(RuntimeError):
    pass


class RequestTooLargeError(LLMError):
    """The prompt exceeds the provider's per-request / per-minute token limit — make it smaller."""


# Shared throttling per endpoint so parallel jobs do not flood the provider.
_throttle: Dict[str, Tuple[threading.Semaphore, list, threading.Lock]] = {}
_throttle_lock = threading.Lock()


def _get_throttle(key: str, concurrency: int):
    with _throttle_lock:
        if key not in _throttle:
            _throttle[key] = (threading.Semaphore(max(1, concurrency)), [0.0], threading.Lock())
        return _throttle[key]


def load_prompt(prompts_dir: Path, name: str) -> Tuple[str, str]:
    """Prompt files contain a '### SYSTEM' section and a '### USER' section."""
    text = (prompts_dir / name).read_text(encoding="utf-8")
    if "### USER" not in text:
        raise LLMError(f"Prompt file {name} must contain '### SYSTEM' and '### USER' sections")
    system, user = text.split("### USER", 1)
    system = system.replace("### SYSTEM", "", 1).strip()
    return system, user.strip()


def fill_template(template: str, **values: Any) -> str:
    for key, val in values.items():
        template = template.replace("{{" + key + "}}", str(val))
    return template


def parse_json_loose(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences, prose and minor syntax errors."""
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = text.strip()
    # Remove <think>...</think> blocks emitted by some reasoning models
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1], default=-1)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = candidate
    try:
        from json_repair import repair_json  # optional dependency

        repaired = repair_json(cleaned, return_objects=True)
        if repaired not in ("", None, {}, []):
            return repaired
    except ImportError:
        pass
    raise ValueError("response is not valid JSON")


class LLMClient:
    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self.settings = settings
        self.log = logger or logging.getLogger("shorts.llm")
        preset = PROVIDERS[settings.llm_provider]
        self.provider = settings.llm_provider
        self.base_url = settings.llm_base_url or preset["base_url"]
        self.model = settings.llm_model or preset["model"]
        if not self.base_url:
            raise LLMError("LLM_BASE_URL is required for the 'custom' provider")
        if not self.model:
            raise LLMError("LLM_MODEL is required for the 'custom' provider")

        key = settings.llm_api_key
        if not key and preset["key_field"]:
            key = getattr(settings, preset["key_field"], "")
        if not key:
            if self.provider == "ollama":
                key = "ollama"
            else:
                env = (preset["key_field"] or "llm_api_key").upper()
                raise LLMError(f"No API key configured for LLM provider '{self.provider}'. Set {env} in .env.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from exc

        keys = [key]
        if self.provider == "groq" and not settings.llm_api_key:
            keys = settings.groq_keys or [key]
        self._pool = get_key_pool(f"llm {self.provider}", keys)
        self._OpenAI = OpenAI
        self._clients: Dict[str, Any] = {}
        # with several keys a 429 should switch keys at once, not be retried on the same key by the SDK
        self._sdk_retries = 0 if len(self._pool) > 1 else 4
        self.client = self._client_for(self._pool.keys[0])
        self._json_mode = bool(preset["json_mode"])
        self._send_temperature = settings.llm_temperature is not None
        self._sem, self._last, self._last_lock = _get_throttle(
            f"{self.provider}|{self.base_url}", settings.llm_max_concurrency
        )

    def fits(self, system: str, user: str) -> bool:
        """True when the prompt is within the per-request token budget (always True when no budget)."""
        budget = self.settings.effective_request_budget
        return not budget or estimate_tokens(system) + estimate_tokens(user) <= budget


    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    # ---------------------------------------------------------------- low level
    def _throttle_wait(self) -> None:
        with self._last_lock:
            delta = time.time() - self._last[0]
            if delta < self.settings.llm_min_interval:
                time.sleep(self.settings.llm_min_interval - delta)
            self._last[0] = time.time()

    def _client_for(self, key: str) -> Any:
        if key not in self._clients:
            self._clients[key] = self._OpenAI(api_key=key, base_url=self.base_url, max_retries=self._sdk_retries,
                                              timeout=240)
        return self._clients[key]

    def _tpm_limiter(self, key: str):
        from core.rate_limiter import get_limiter

        tag = str(self._pool.keys.index(key)) if key in self._pool.keys else "x"
        return get_limiter(f"llm-tpm {self.provider} {tag}", None, [(60.0, float(self.settings.effective_llm_tpm))])

    def _send(self, messages: List[Dict[str, str]], cancel_event: Optional[threading.Event] = None) -> str:
        """One completion, rotating keys on rate limits / rejected keys."""
        import openai

        tpm = self.settings.effective_llm_tpm
        units = min(sum(estimate_tokens(m["content"]) for m in messages) + 800, tpm) if tpm else 0
        last: Optional[BaseException] = None
        for _ in range(len(self._pool) * 2 + 1):
            key, wait = self._pool.pick((lambda k: self._tpm_limiter(k).wait_time(units)) if tpm else None)
            if wait > 0:
                if wait > 3:
                    self.log.info("All %d %s key(s) are rate-limited — waiting %.0fs", len(self._pool),
                                  self.provider, wait)
                interruptible_sleep(min(wait, 90), cancel_event)
            if tpm:
                self._tpm_limiter(key).acquire(units, cancel_event, self.log)
            try:
                return self._complete(messages, self._client_for(key))
            except openai.RateLimitError as exc:
                last = exc
                delay = _retry_after(exc, default=20.0)
                if "request too large" in str(exc).lower():
                    raise
                if self._pool.cooldown(key, delay):
                    nxt, _w = self._pool.pick()
                    self.log.warning("%s %s rate-limited (%.0fs) — switching to %s", self.provider,
                                     self._pool.label(key), delay, self._pool.label(nxt))
                    continue
                raise
            except openai.AuthenticationError as exc:
                last = exc
                if len(self._pool) > 1 and self._pool.invalid(key, "rejected by the provider (invalid key?)"):
                    self.log.warning("%s %s was rejected — skipping it", self.provider, self._pool.label(key))
                    continue
                raise
        if last:
            raise last
        raise LLMError("LLM request could not be completed")

    def _complete(self, messages: list, client: Any = None) -> str:
        import openai

        client = client or self.client

        with self._sem:
            for _ in range(3):  # adapt to provider quirks (unsupported params)
                self._throttle_wait()
                kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
                if self._json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self._send_temperature:
                    kwargs["temperature"] = self.settings.llm_temperature
                try:
                    resp = client.chat.completions.create(**kwargs)
                except openai.BadRequestError as exc:
                    msg = str(exc).lower()
                    if self._send_temperature and "temperature" in msg:
                        self.log.info("Model %s rejects 'temperature'; retrying without it", self.model)
                        self._send_temperature = False
                        continue
                    if self._json_mode and ("response_format" in msg or "json" in msg):
                        self.log.info("Model %s rejects JSON mode; retrying without it", self.model)
                        self._json_mode = False
                        continue
                    raise
                if not resp.choices:
                    raise ValueError("LLM returned no choices")
                return resp.choices[0].message.content or ""
        raise LLMError("LLM request could not be completed")

    # ---------------------------------------------------------------- public
    def chat_json(
        self,
        system: str,
        user: str,
        validate: Optional[Callable[[Any], T]] = None,
        attempts: int = 3,
        what: str = "LLM request",
        cancel_event: Optional[threading.Event] = None,
    ) -> T:
        """Send a prompt, parse JSON, validate. Invalid output is fed back for repair."""
        import openai

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled("Job cancelled by user")
            try:
                content = self._send(messages, cancel_event)
            except (openai.AuthenticationError, NoValidKeyError) as exc:
                raise LLMError(f"LLM authentication failed for {self.label}: check your API key(s)") from exc
            except openai.NotFoundError as exc:
                raise LLMError(f"LLM model or endpoint not found ({self.label}): {exc}") from exc
            except openai.RateLimitError as exc:
                last_error = exc
                delay = _retry_after(exc, default=20.0 * attempt)
                self.log.warning("%s: rate limited by %s; waiting %.0fs", what, self.provider, delay)
                interruptible_sleep(delay, cancel_event)
                continue
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as exc:
                last_error = exc
                delay = 10 * attempt
                self.log.warning("%s: transient LLM error (%s); retrying in %ss", what, type(exc).__name__, delay)
                interruptible_sleep(delay, cancel_event)
                continue
            except openai.APIStatusError as exc:
                text = str(exc).lower()
                if exc.status_code == 413 or "request too large" in text or "reduce your message size" in text:
                    raise RequestTooLargeError(f"{what}: request too large for {self.label} ({exc})") from exc
                last_error = exc
                self.log.warning("%s: LLM API error %s", what, exc)
                interruptible_sleep(5 * attempt, cancel_event)
                continue

            try:
                data = parse_json_loose(content)
                return validate(data) if validate else data  # type: ignore[return-value]
            except Exception as exc:  # invalid JSON / schema
                last_error = exc
                self.log.warning("%s: invalid JSON from LLM (attempt %d/%d): %s", what, attempt, attempts, exc)
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:6000]},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply could not be used: {str(exc)[:500]}\n"
                            "Reply again with ONLY one valid JSON object that follows the required schema exactly. "
                            "No markdown, no comments, no trailing commas."
                        ),
                    },
                ]
        raise LLMError(f"{what} failed after {attempts} attempts: {last_error}")

    def ping(self) -> str:
        data = self.chat_json(
            "You are a health-check endpoint. Reply with JSON only.",
            'Return exactly {"ok": true}',
            what="LLM ping",
            attempts=2,
        )
        return json.dumps(data)
