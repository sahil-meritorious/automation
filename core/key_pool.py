"""API-key rotation: when one key is rate-limited, switch to the next one immediately.

* Keys are tried in order; a key that gets HTTP 429 is put on cooldown for the time the
  provider asks (retry-after), and the request moves to the next available key.
* Only when every key is cooling down does the caller wait — for the shortest cooldown.
* Invalid keys (401/403) are parked for 24 h so a typo never blocks the others.
* Shared process-wide, so parallel jobs and threads rotate through the same pool.
Note: keys in the same Groq organisation share one set of limits; rotation only helps
when the keys belong to different organisations.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("shorts.keys")


class NoValidKeyError(RuntimeError):
    pass


class KeyPool:
    def __init__(self, name: str, keys: List[str]) -> None:
        self.name = name
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k and k.strip()))
        self._cool: Dict[str, float] = {}
        self._dead: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._idx = 0

    def __len__(self) -> int:
        return len(self.keys)

    def label(self, key: str) -> str:
        try:
            return f"key {self.keys.index(key) + 1}/{len(self.keys)}"
        except ValueError:
            return "key ?"

    def pick(self, extra_wait: Optional[Callable[[str], float]] = None) -> Tuple[str, float]:
        """(key, seconds to wait). Prefers the current key, then the next ones in order."""
        with self._lock:
            live = [k for k in self.keys if k not in self._dead]
            if not live:
                raise NoValidKeyError(f"No valid {self.name} API key left: " + "; ".join(self._dead.values()))
            now = time.time()
            order = live[self._idx % len(live):] + live[: self._idx % len(live)]
            best, best_wait = order[0], float("inf")
            for k in order:
                wait = max(0.0, self._cool.get(k, 0.0) - now)
                if extra_wait is not None:
                    wait = max(wait, extra_wait(k))
                if wait <= 0:
                    self._idx = live.index(k)
                    return k, 0.0
                if wait < best_wait:
                    best, best_wait = k, wait
            self._idx = live.index(best)
            return best, best_wait

    def cooldown(self, key: str, seconds: float) -> bool:
        """Mark a key as rate-limited. Returns True when another key is available right now."""
        with self._lock:
            self._cool[key] = time.time() + max(1.0, seconds)
            if key in self.keys and len(self.keys) > 1:
                self._idx = (self.keys.index(key) + 1) % len(self.keys)
            now = time.time()
            return any(self._cool.get(k, 0) <= now for k in self.keys if k not in self._dead)

    def invalid(self, key: str, reason: str) -> bool:
        """Park a rejected key. Returns True when other usable keys remain."""
        with self._lock:
            self._dead[key] = f"{self.label(key)}: {reason}"
            return any(k not in self._dead for k in self.keys)


_pools: Dict[str, KeyPool] = {}
_pools_lock = threading.Lock()


def get_key_pool(name: str, keys: List[str]) -> KeyPool:
    key = name + "|" + "|".join(keys)
    with _pools_lock:
        if key not in _pools:
            _pools[key] = KeyPool(name, keys)
        return _pools[key]
