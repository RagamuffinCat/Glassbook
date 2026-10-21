"""Public trade stream: find native TWAP slices and the wallets behind them.

Native TWAP slices have an all-zero `hash`, and each trade carries `users: [buyer, seller]`,
so TWAP wallets can be found without knowing any address.
"""

import json
from collections import defaultdict
from pathlib import Path

from .client import default_client

ZERO_HASH_PREFIX = "0x" + "0" * 40


def post_info(payload, timeout=None, retries=None):
    c = default_client()
    if timeout is not None:
        c.timeout = float(timeout)      # sticks for all later callers (shared client)
    return c.post(payload, retries=retries)


def is_twap_slice(trade):
    h = trade.get("hash") or ""
    return h.startswith(ZERO_HASH_PREFIX)


def analyze_stream(path):
    """Summarize a raw trade-stream file (one trade per line).

    Only one side of a zero-hash trade is the TWAP; the other is a resting order.
    Both sides become candidates, ranked with the side-inferred taker first.
    The feasibility check later tests whether that inference holds.
    """
    total = 0
    with_users = 0
    twap = 0
    twap_notional = 0.0
    total_notional = 0.0
    all_wallets = set()
    twap_wallets = defaultdict(int)      # Both sides
    twap_aggressor = defaultdict(int)    # Inferred taker only
    per_coin = defaultdict(lambda: [0, 0])  # [all, twap]

    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            coin = t.get("coin", "?")
            per_coin[coin][0] += 1
            try:
                notional = float(t.get("px", 0)) * float(t.get("sz", 0))
            except (TypeError, ValueError):
                notional = 0.0
            total_notional += notional

            users = t.get("users") or []
            users = [u for u in users if isinstance(u, str) and u.startswith("0x")]
            if len(users) >= 2:
                with_users += 1
                all_wallets.update(users)

            if is_twap_slice(t):
                twap += 1
                twap_notional += notional
                per_coin[coin][1] += 1
                for u in users:
                    twap_wallets[u] += 1
                if len(users) >= 2:
                    # Assumes users == [buyer, seller] and side "B" means the buyer took
                    twap_aggressor[users[0] if t.get("side") == "B" else users[1]] += 1

    ranking = sorted(
        twap_wallets.items(),
        key=lambda kv: (twap_aggressor.get(kv[0], 0), kv[1]),
        reverse=True,
    )

    return {
        "trades_total": total,
        "trades_with_users": with_users,
        "user_attribution_ratio": (with_users / total) if total else 0.0,
        "twap_slices": twap,
        "twap_share_of_trades": (twap / total) if total else 0.0,
        "twap_share_of_notional": (twap_notional / total_notional) if total_notional else 0.0,
        "notional_usd": round(total_notional, 2),
        "unique_wallets": len(all_wallets),
        "twap_candidate_wallets": len(twap_wallets),
        "twap_aggressor_wallets": len(twap_aggressor),
        "twap_wallet_ranking": ranking,
        "aggressor_set": sorted(twap_aggressor),
        "per_coin": {k: {"trades": v[0], "twap": v[1]} for k, v in
                     sorted(per_coin.items(), key=lambda kv: -kv[1][0])},
    }
