"""Per-client token bucket.

In-process and therefore per-worker, which is the right trade for a single free
tier instance. Move to Redis or a Durable Object the moment you run more than
one process, or the effective limit silently multiplies by the worker count.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple


class TokenBucket:
    def __init__(self, capacity: int = 20, refill_per_sec: float = 0.2):
        self.capacity = capacity
        self.refill = refill_per_sec
        self._state: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= cost:
                self._state[key] = (tokens - cost, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            if self.refill <= 0:
                # A zero refill rate is a hard quota with no recovery, so there
                # is no finite retry-after to report.
                return False, float("inf")
            return False, (cost - tokens) / self.refill

    def sweep(self, max_entries: int = 10000) -> None:
        """Drop the oldest entries so a scraper cannot grow this without bound."""
        with self._lock:
            if len(self._state) <= max_entries:
                return
            for key in sorted(self._state, key=lambda k: self._state[k][1])[: len(self._state) // 2]:
                del self._state[key]
