"""Thread-safe sliding-window rate limiter (requests/minute + usage units per window)."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from core.utils import JobCancelled

log = logging.getLogger("shorts.ratelimit")


class RateLimiter:
    """Blocks callers until a request fits inside every configured limit.

    * ``requests_per_minute`` — classic RPM limit.
    * ``unit_limits`` — list of ``(window_seconds, max_units)``; used for Groq's
      "audio seconds per hour/day" limits.
    * :meth:`penalize` pauses everybody (used after an HTTP 429).
    """

    def __init__(
        self,
        name: str,
        requests_per_minute: Optional[int] = None,
        unit_limits: Sequence[Tuple[float, float]] = (),
    ) -> None:
        self.name = name
        self.rpm = requests_per_minute if requests_per_minute and requests_per_minute > 0 else None
        self.unit_limits = [(float(w), float(m)) for w, m in unit_limits if m and m > 0]
        self._lock = threading.Lock()
        self._requests: Deque[float] = deque()
        self._units: Deque[Tuple[float, float]] = deque()
        self._blocked_until = 0.0
        self._max_window = max([60.0] + [w for w, _ in self.unit_limits])

    def _wait_time(self, now: float, units: float) -> float:
        if now < self._blocked_until:
            return self._blocked_until - now
        while self._requests and self._requests[0] <= now - 60:
            self._requests.popleft()
        while self._units and self._units[0][0] <= now - self._max_window:
            self._units.popleft()
        wait = 0.0
        if self.rpm and len(self._requests) >= self.rpm:
            wait = max(wait, self._requests[0] + 60 - now)
        for window, limit in self.unit_limits:
            inside = [(t, u) for t, u in self._units if t > now - window]
            total = sum(u for _, u in inside)
            if total > 0 and total + units > limit:
                excess = total + units - limit
                freed = 0.0
                for t, u in inside:
                    freed += u
                    if freed >= excess:
                        wait = max(wait, t + window - now)
                        break
                else:
                    wait = max(wait, inside[-1][0] + window - now)
        return max(0.0, wait)

    def acquire(
        self,
        units: float = 0.0,
        cancel_event: Optional[threading.Event] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logger = logger or log
        announced = False
        while True:
            with self._lock:
                now = time.time()
                wait = self._wait_time(now, units)
                if wait <= 0:
                    self._requests.append(now)
                    if units:
                        self._units.append((now, units))
                    return
            if not announced and wait > 3:
                logger.info("[%s] rate limit: waiting %.0fs before next request", self.name, wait)
                announced = True
            step = min(wait, 5.0)
            if cancel_event is not None:
                if cancel_event.wait(step):
                    raise JobCancelled("Job cancelled by user")
            else:
                time.sleep(step)

    def wait_time(self, units: float = 0.0) -> float:
        """Seconds until a request of ``units`` would be allowed (0 = now)."""
        with self._lock:
            return self._wait_time(time.time(), units)

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.time() + max(0.0, seconds))


_registry: Dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_limiter(name: str, requests_per_minute: Optional[int], unit_limits: List[Tuple[float, float]]) -> RateLimiter:
    """Process-wide shared limiter so parallel jobs respect the same API quota."""
    key = f"{name}|{requests_per_minute}|{unit_limits}"
    with _registry_lock:
        if key not in _registry:
            _registry[key] = RateLimiter(name, requests_per_minute, unit_limits)
        return _registry[key]
