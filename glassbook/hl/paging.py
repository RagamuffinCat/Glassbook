"""Time-cursor pagination for the info API's *ByTime endpoints (2000 items per page)."""

import time

from .client import APIRequestFailed

PAGE = 2000
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# Full history means startTime=0. Empty windows double in size, so walking back to 1970
# costs only about a dozen extra empty requests (weight 20 each).
WINDOW_JUMP_MS = 30 * DAY_MS             # Window size once the 1h probe has succeeded
MIN_WINDOW_MS = HOUR_MS                  # A failure at this size fails the wallet
DEFAULT_MAX_PAGES = 600                  # Per wallet, 600 x 2000 = 1.2M fills
COOLDOWN_S = 5.0


def fill_tid(item) -> int | None:
    """Trade id of an API item, which is either a fill or {"fill": ..., "twapId": ...}."""
    f = item.get("fill", item) if isinstance(item, dict) else {}
    v = f.get("tid")
    return int(v) if v is not None else None


def fill_time(item) -> int:
    f = item.get("fill", item) if isinstance(item, dict) else {}
    return int(f.get("time", 0))


def _paginate_window(fetch, user: str, start_ms: int, end_ms: int,
                     seen: set[int], out: list, page: int, max_pages: int) -> dict:
    """Page [start_ms, end_ms] with a startTime cursor, de-duplicating by tid.

    `seen` and `out` belong to the caller, so pages fetched before a failure are kept and
    a retry does not double them.
    """
    cursor, pages, forced, deduped, no_tid, kept = start_ms, 0, 0, 0, 0, 0
    hit_max = False

    while cursor <= end_ms:
        if pages >= max_pages:
            hit_max = True
            break
        batch = fetch(user, cursor, end_ms)
        pages += 1
        # print(user[:10], cursor, len(batch))
        if not batch:
            break
        new = []
        for b in batch:
            t = fill_tid(b)
            if t is None:
                no_tid += 1
                continue
            if t in seen:
                deduped += 1
                continue
            seen.add(t)
            new.append(b)
        out.extend(new)
        kept += len(new)
        if len(batch) < page:
            break
        last_t = max(fill_time(b) for b in batch)
        # The cursor is inclusive, so the next page starts with the fills at last_t again (dropped
        # by tid above). A page that brings nothing new pushes the cursor 1 ms forward.
        # TODO >2000 fills in one ms -> rest of that ms skipped. not seen yet
        if not new or last_t <= cursor:
            cursor = max(last_t, cursor) + 1
            forced += 1
        else:
            cursor = last_t

    return {"pages": pages, "forced_advances": forced, "hit_max_pages": hit_max,
            "deduped": deduped, "items_without_tid": no_tid, "kept": kept,
            "dedup_suspicious": deduped > page * max(1, forced + 1)}


def paginate(fetch, user, start_ms, end_ms, page=PAGE, max_pages=400):
    """Single-window pagination, used by the pagination check."""
    out = []
    meta = _paginate_window(fetch, user, start_ms, end_ms, set(), out, page, max_pages)
    return out, meta


def paginate_adaptive(fetch, user, start_ms, end_ms, page=PAGE,
                      max_pages=DEFAULT_MAX_PAGES, jump_ms=WINDOW_JUMP_MS,
                      min_window_ms=MIN_WINDOW_MS):
    """Paginate in time windows walking back from `end_ms`.

    Heavy targeted wallets time out on a full-history request. Server cost probably grows with
    fills per window (unconfirmed, light wallets have timed out too).
      first window = whole span, so light wallets still take 1 request
      1st failure -> probe 1h. 1h fails too -> smaller windows won't help
      else 30-day windows, halve on failure (floor 1h), never above the last failed size
      empty windows double freely
    """
    seen = set()
    out = []
    calls = [0]

    def counted(u, s, e):
        calls[0] += 1
        return fetch(u, s, e)

    span = end_ms - start_ms + 1
    window, cap = span, span
    hi = end_ms
    probed = False
    meta = {"windows": 0, "pages": 0, "forced_advances": 0, "deduped": 0,
            "items_without_tid": 0, "failures": 0, "failed_pages": 0, "splits": 0,
            "probe": None, "min_window_ms": span, "hit_max_pages": False}

    def run_window(lo, hi_):
        c0 = calls[0]
        try:
            pm = _paginate_window(counted, user, lo, hi_, seen, out, page,
                                  max_pages - calls[0])
        except APIRequestFailed:
            meta["failed_pages"] += calls[0] - c0
            meta["failures"] += 1
            raise
        meta["windows"] += 1
        for k in ("pages", "forced_advances", "deduped", "items_without_tid"):
            meta[k] += pm[k]
        meta["min_window_ms"] = min(meta["min_window_ms"], hi_ - lo + 1)
        return pm

    while hi >= start_ms:
        if calls[0] >= max_pages:
            meta["hit_max_pages"] = True
            break
        lo = max(start_ms, hi - window + 1)
        # print(f"{user[:10]} window {(hi - lo) / HOUR_MS:.0f}h cap {cap / HOUR_MS:.0f}h, {len(out)} so far")
        try:
            pm = run_window(lo, hi)
        except APIRequestFailed as e:
            time.sleep(COOLDOWN_S)
            if not probed:
                probed = True
                plo = max(start_ms, hi - min_window_ms + 1)
                try:
                    pm = run_window(plo, hi)
                except APIRequestFailed as e2:
                    meta["probe"] = "failed"
                    raise APIRequestFailed(f"1h probe window failed too: {e2}") from e2
                meta["probe"] = "ok"
                meta["splits"] += 1
                hi = plo - 1
                window = cap = min(jump_ms, span)
                continue
            if window <= min_window_ms:
                raise APIRequestFailed(f"still failing at {window / HOUR_MS:.1f}h windows: {e}") from e
            window = cap = max(min_window_ms, window // 2)
            meta["splits"] += 1
            continue
        if pm["hit_max_pages"]:
            meta["hit_max_pages"] = True
            break
        hi = lo - 1
        if pm["kept"] == 0:
            window = min(window * 2, span)
        elif pm["pages"] <= 1 and pm["kept"] < page // 2:
            window = min(window * 2, cap)               # Under half a page

    meta["kept"] = len(out)
    meta["calls"] = calls[0]
    # dups expected from refetches after a failure, window edges, forced advances.
    # way more than that -> cursor bug
    meta["dedup_suspicious"] = meta["deduped"] > page * (
        meta["failed_pages"] + meta["windows"] + meta["forced_advances"] + 1)
    return out, meta
