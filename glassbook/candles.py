"""Candle intervals and file names, in one place so collect_candles.py and the impact code agree."""

from pathlib import Path

MIN_MS = 60_000
INTERVALS = [("1m", MIN_MS), ("5m", 5 * MIN_MS), ("15m", 15 * MIN_MS),
             ("1h", 60 * MIN_MS), ("4h", 240 * MIN_MS), ("1d", 1440 * MIN_MS)]


def choose_interval(need_start_ms: int, retention: dict[str, int | None]) -> str | None:
    """Finest interval whose earliest bar (retention[interval], ms) is at or before need_start_ms, else None."""
    for name, _ in INTERVALS:
        r = retention.get(name)
        if r is not None and r <= need_start_ms:
            return name
    return None


def safe_name(coin: str) -> str:
    """File-safe coin name for things like PURR/USDC or @107.

    Non-alphanumerics -> "_", plus a char-code sum so "A/B" and "A_B" differ.
    Not hash(): salted per process, names would change between runs.
    """
    return "".join(ch if ch.isalnum() else "_" for ch in coin) + "_" + str(sum(ord(c) for c in coin))


def candle_path(cdir: Path, coin: str, interval: str) -> Path:
    return Path(cdir) / f"{safe_name(coin)}_{interval}.jsonl"
