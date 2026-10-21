"""Hyperliquid info API client with weight-based rate limiting.

    from glassbook.hl.client import default_client
    mids = default_client().post({"type": "allMids"})
"""

import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections import deque

API = "https://api.hyperliquid.xyz/info"
# Official limit: 1200 weight/min per IP, shared by all REST requests. A fixed 1.1 s between requests
# looks safe but comes to about 6500 weight/min on full pages, 5x the limit.
BUDGET_PER_MIN = 1200
SAFETY = 0.75  # Other processes on the same IP, and our estimates may be off
MAX_ITEMS = 2000

# Weights from the docs: LIGHT 2, userRole 60, anything else 20. PER_ITEM types add 1 per 20 items
# returned, so a full page costs 120.
LIGHT = {"l2Book", "allMids", "clearinghouseState", "orderStatus",
         "spotClearinghouseState", "exchangeStatus"}
PER_ITEM = {"userFills", "userFillsByTime", "userTwapSliceFills",
            "userTwapSliceFillsByTime", "fundingHistory", "userFunding",
            "nonUserFundingUpdates", "twapHistory",
            "candleSnapshot"}  # Not documented as per-item, reserve the full-page cost anyway


class APIRequestFailed(RuntimeError):
    pass


def base_weight(t: str) -> int:
    if t in LIGHT:
        return 2
    if t == "userRole":
        return 60
    return 20


def worst_weight(t: str) -> int:
    return base_weight(t) + (MAX_ITEMS // 20 if t in PER_ITEM else 0)


def actual_weight(t: str, resp) -> int:
    extra = math.ceil(len(resp) / 20) if (t in PER_ITEM and isinstance(resp, list)) else 0
    return base_weight(t) + extra


class RateLimitedClient:
    def __init__(self, budget: float = BUDGET_PER_MIN * SAFETY, timeout: float = 45.0,
                 backoff: tuple[float, ...] = (5.0, 15.0, 45.0)):
        self.budget = budget
        self.timeout = timeout
        self.backoff = backoff
        self.window: deque[tuple[float, int]] = deque()   # (time, weight) of the last 60 s
        self.stats = {"calls": 0, "weight": 0, "waited_s": 0.0,
                      "timeouts": 0, "retries": 0, "failed": 0}

    def _used(self, now: float) -> int:
        while self.window and now - self.window[0][0] >= 60.0:
            self.window.popleft()
        return sum(w for _, w in self.window)

    def _reserve(self, need: int) -> None:
        need = min(need, int(self.budget))   # else need > budget never fits
        while True:
            now = time.monotonic()
            if self._used(now) + need <= self.budget:
                return
            wait = max(60.0 - (now - self.window[0][0]) + 0.05, 0.05)
            self.stats["waited_s"] += wait
            time.sleep(wait)

    def used_now(self) -> int:
        return self._used(time.monotonic())

    def _http(self, payload: dict):
        req = urllib.request.Request(
            API, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "glassbook/1.0"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout,
                                    context=ssl.create_default_context()) as r:
            return json.loads(r.read().decode())

    def post(self, payload: dict, retries: int | None = None):
        t = payload.get("type", "")
        tries = (len(self.backoff) + 1) if retries is None else retries
        last: Exception | None = None

        for attempt in range(tries):
            # Reserve the worst case, then book what the response actually cost
            self._reserve(worst_weight(t))
            self.stats["calls"] += 1
            try:
                resp = self._http(payload)
            except (TimeoutError, socket.timeout) as e:
                last = e
                self.stats["timeouts"] += 1
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429:
                    self.stats["timeouts"] += 1
                elif 400 <= e.code < 500:
                    self.window.append((time.monotonic(), base_weight(t)))
                    self.stats["failed"] += 1
                    raise APIRequestFailed(f"{t}: HTTP {e.code}") from e
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
                last = e
            else:
                w = actual_weight(t, resp)
                self.window.append((time.monotonic(), w))
                self.stats["weight"] += w
                return resp

            # unclear if failed attempts are billed, book them anyway
            self.window.append((time.monotonic(), base_weight(t)))
            if attempt < tries - 1:
                self.stats["retries"] += 1
                time.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])

        self.stats["failed"] += 1
        raise APIRequestFailed(f"{t}: {type(last).__name__}: {str(last)[:100]}")

    def preflight(self) -> float:
        """allMids (weight 2) before a long run: fail now, not an hour in. Returns latency in s."""
        t0 = time.monotonic()
        r = self.post({"type": "allMids"}, retries=2)
        if not isinstance(r, dict) or not r:
            raise APIRequestFailed(f"allMids: unexpected response {str(r)[:120]}")
        return round(time.monotonic() - t0, 2)


# one client per process = one shared budget
_default: RateLimitedClient | None = None


def default_client() -> RateLimitedClient:
    global _default
    if _default is None:
        _default = RateLimitedClient()
    return _default
