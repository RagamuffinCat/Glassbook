#!/usr/bin/env python3
"""Feasibility check: does Hyperliquid's public data carry usable metaorder labels?

The public `trades` WebSocket stream gives `users: [buyer, seller]`, and native TWAP
slices have an all-zero `hash`, so TWAP wallets can be found without knowing any
address. For those wallets `userTwapSliceFills` returns fills labeled by `twapId`.
Public endpoints only, about five minutes.

Passes if at least 99% of trades carry `users`, TWAP slices are at least 0.5% of all trades,
at least 30 metaorders come back and the median metaorder has 5 slices or more.

Needs `pip install websockets`.
    python3 scripts/check_feasibility.py                  # collect for 300 s
    python3 scripts/check_feasibility.py --seconds 600    # longer window, more stable numbers
    python3 scripts/check_feasibility.py --analyze-only data/raw/trades_<stamp>.jsonl

Writes data/raw/trades_<stamp>.jsonl (the raw stream; later steps read it),
data/raw/twapfills_<stamp>.json and the result in data/feasibility_report.json.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CHECKS = {
    "user_attribution": {
        "desc": "Share of trades carrying the users field",
        "threshold": 0.99,
        "unit": "ratio",
    },
    "twap_share": {
        "desc": "Native TWAP slices as a share of all trades",
        "threshold": 0.005,
        "unit": "ratio",
    },
    "metaorders": {
        "desc": "Distinct twapId metaorders recovered from the sampled wallets",
        "threshold": 30,
        "unit": "count",
    },
    "slices_per_metaorder": {
        "desc": "Median slices per metaorder",
        "threshold": 5,
        "unit": "count",
    },
}

WS = "wss://api.hyperliquid.xyz/ws"

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "SUI",
    "AVAX", "LTC", "BNB", "ARB", "OP", "LINK", "AAVE", "PEPE",
]

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
RAW = ROOT / "data" / "raw"

from glassbook.hl.stream import analyze_stream, post_info


def resolve_coins(limit):
    """Top perps by open-interest notional, or DEFAULT_COINS if that lookup fails."""
    try:
        meta = post_info({"type": "metaAndAssetCtxs"})
        universe = meta[0]["universe"]
        ctxs = meta[1]
        rows = []
        for u, c in zip(universe, ctxs):
            if u.get("isDelisted"):
                continue
            try:
                oi = float(c.get("openInterest", 0)) * float(c.get("markPx", 0))
            except (TypeError, ValueError):
                oi = 0.0
            rows.append((oi, u["name"]))
        rows.sort(reverse=True)
        coins = [n for _, n in rows[:limit]]
        if coins:
            print(f"  top {len(coins)} perps by open interest: {', '.join(coins[:8])} ...")
            return coins
    except Exception as e:
        print(f"  metaAndAssetCtxs failed ({e}), using the default list")
    return DEFAULT_COINS[:limit]


async def _reader(ws, fh, seconds, counter):
    """Write trades to fh until the task is cancelled. Cancel the task, not wait_for per recv
    (simpler on websockets >= 14)."""
    started = time.time()
    last_report = started
    async for msg in ws:
        try:
            obj = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            continue
        if obj.get("channel") != "trades":
            continue
        for t in obj.get("data", []):
            fh.write(json.dumps(t, separators=(",", ":")) + "\n")
            counter[0] += 1
        now = time.time()
        if now - last_report >= 20:
            fh.flush()
            print(f"    [{int(now - started):>4}s/{seconds}s] {counter[0]:,} trades received", flush=True)
            last_report = now


async def collect(coins, seconds, out_path):
    try:
        import websockets
    except ImportError:
        sys.exit("Missing dependency: pip install websockets")

    counter = [0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(WS, ping_interval=20, ping_timeout=20,
                                  max_size=8 << 20) as ws:
        for c in coins:
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": c},
            }))
        print(f"  subscribed to {len(coins)} perps, {seconds}s")

        with out_path.open("a", buffering=1 << 16) as fh:
            task = asyncio.ensure_future(_reader(ws, fh, seconds, counter))
            done, pending = await asyncio.wait({task}, timeout=seconds)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            for t in done:
                exc = t.exception()
                if exc is not None:
                    print(f"  stream ended early: {type(exc).__name__}: {exc}")
    return counter[0]


def fetch_labeled_metaorders(wallets, out_path, aggressor_set=None, pause=1.1):
    """Fetch userTwapSliceFills per wallet and group fills by twapId.

    Also tests the users order: if inferred takers return data more often than
    the other wallets, the [buyer, seller] convention holds.
    """
    # TODO is the pause still needed? post_info is rate limited
    aggressor_set = aggressor_set or set()
    raw = {}
    groups = defaultdict(list)
    hits = {"aggressor": [0, 0], "other": [0, 0]}  # [with data, queried]

    for i, w in enumerate(wallets, 1):
        try:
            fills = post_info({"type": "userTwapSliceFills", "user": w})
        except Exception as e:
            print(f"    [{i}/{len(wallets)}] {w[:10]}... failed: {e}")
            time.sleep(pause)
            continue
        if not isinstance(fills, list):
            fills = []
        raw[w] = fills
        bucket = "aggressor" if w in aggressor_set else "other"
        hits[bucket][1] += 1
        if fills:
            hits[bucket][0] += 1
        for item in fills:
            # Items are {"fill": {...}, "twapId": n}, but take a flat fill too
            fill = item.get("fill", item) if isinstance(item, dict) else {}
            tid = item.get("twapId") if isinstance(item, dict) else None
            if tid is None:
                tid = fill.get("twapId")
            if tid is None:
                continue
            groups[(w, int(tid))].append(fill)
        print(f"    [{i}/{len(wallets)}] {w[:10]}... {len(fills):>5} TWAP slices", flush=True)
        time.sleep(pause)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=1))

    metas = []
    for (wallet, tid), fills in groups.items():
        if not fills:
            continue
        times, notionals, szs = [], [], []
        for fl in fills:
            try:
                px, sz = float(fl.get("px", 0)), float(fl.get("sz", 0))
            except (TypeError, ValueError):
                px = sz = 0.0
            notionals.append(px * sz)
            szs.append(sz)
            if fl.get("time") is not None:
                times.append(int(fl["time"]))
        metas.append({
            "wallet": wallet,
            "twap_id": tid,
            "coin": fills[0].get("coin"),
            "side": fills[0].get("side"),
            "slices": len(fills),
            "size": round(sum(szs), 6),
            "notional_usd": round(sum(notionals), 2),
            "duration_s": round((max(times) - min(times)) / 1000, 1) if len(times) > 1 else 0.0,
            "t_start_ms": min(times) if times else None,
        })

    slices = [m["slices"] for m in metas]
    notionals = [m["notional_usd"] for m in metas if m["notional_usd"] > 0]
    durations = [m["duration_s"] for m in metas if m["duration_s"] > 0]

    def q(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(p * len(s)))], 2)

    def rate(b):
        return round(hits[b][0] / hits[b][1], 3) if hits[b][1] else None

    return {
        "wallets_queried": len(wallets),
        "wallets_with_data": sum(1 for v in raw.values() if v),
        "convention_check": {
            "aggressor_hit_rate": rate("aggressor"),
            "other_hit_rate": rate("other"),
            "n_aggressor": hits["aggressor"][1],
            "n_other": hits["other"][1],
        },
        "metaorders": len(metas),
        "slices_total": sum(slices),
        "slices_median": statistics.median(slices) if slices else 0,
        "slices_p90": q(slices, 0.9),
        "notional_median_usd": statistics.median(notionals) if notionals else 0.0,
        "notional_p10_usd": q(notionals, 0.10),
        "notional_p90_usd": q(notionals, 0.90),
        "duration_median_s": statistics.median(durations) if durations else 0.0,
        "coins_covered": len({m["coin"] for m in metas}),
        "sample": sorted(metas, key=lambda m: -m["notional_usd"])[:10],
    }


def verdict(stream, labeled):
    observed = {
        "user_attribution": stream["user_attribution_ratio"],
        "twap_share": stream["twap_share_of_trades"],
        "metaorders": labeled["metaorders"],
        "slices_per_metaorder": labeled["slices_median"],
    }
    results = {}
    for k, spec in CHECKS.items():
        v = observed[k]
        results[k] = {
            "desc": spec["desc"],
            "observed": v,
            "threshold": spec["threshold"],
            "pass": v >= spec["threshold"],
        }
    return {"checks": results, "all_pass": all(r["pass"] for r in results.values())}


def fmt(v, unit):
    return f"{v*100:.2f}%" if unit == "ratio" else f"{v:,.0f}"


def report(stream, labeled, v, seconds):
    print(f"\nwindow {seconds}s, {stream['trades_total']:,} trades, ${stream['notional_usd']:,.0f}, "
          f"{stream['unique_wallets']:,} wallets")
    print(f"TWAP slices {stream['twap_slices']:,} ({stream['twap_share_of_trades']*100:.2f}% of trades, "
          f"{stream['twap_share_of_notional']*100:.2f}% of notional), candidates "
          f"{stream['twap_candidate_wallets']:,} ({stream['twap_aggressor_wallets']:,} as taker)")
    print(f"wallets queried {labeled['wallets_queried']}, with data {labeled['wallets_with_data']}")
    cc = labeled.get("convention_check") or {}
    a, o = cc.get("aggressor_hit_rate"), cc.get("other_hit_rate")
    if a is not None and o is not None:
        tag = "holds" if a > o else ("reversed?" if o > a else "can't tell")
        print(f"users order: taker hit {a*100:.0f}% (n={cc.get('n_aggressor', 0)}) vs counterparty "
              f"{o*100:.0f}% (n={cc.get('n_other', 0)}), {tag}")
    else:
        # One-sided sample, nothing to compare against
        print("users order: not verified, sample is one-sided")
    print(f"metaorders {labeled['metaorders']:,} on {labeled['coins_covered']} perps, slices median "
          f"{labeled['slices_median']:.0f} / p90 {labeled['slices_p90']:.0f}, notional median "
          f"${labeled['notional_median_usd']:,.0f}, duration median {labeled['duration_median_s']:,.0f}s")
    for m in labeled["sample"][:6]:
        print(f"    {str(m['coin']):<8}{str(m['side']):<4}{m['slices']:>6} slices {m['notional_usd']:>14,.0f}"
              f"{m['duration_s']:>9,.0f}s")
    print()
    for k, r in v["checks"].items():
        unit = CHECKS[k]["unit"]
        print(f"  {'pass' if r['pass'] else 'FAIL'}  {k:<22}{fmt(r['observed'], unit):>10}  (>= {fmt(r['threshold'], unit)})")
    print()


def main():
    ap = argparse.ArgumentParser(description="Check whether the public data carries usable metaorder labels")
    ap.add_argument("--seconds", type=int, default=300, help="stream collection time in seconds (default 300)")
    ap.add_argument("--coins", type=int, default=15, help="number of perps to subscribe to (default 15)")
    ap.add_argument("--wallets", type=int, default=25, help="number of TWAP wallets to query (default 25)")
    ap.add_argument("--analyze-only", type=str, default=None, metavar="JSONL",
                    help="skip collection and analyze an existing raw jsonl file")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RAW.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        raw_path = Path(args.analyze_only)
        if not raw_path.exists():
            sys.exit(f"Not found: {raw_path}")
        print(f"reading {raw_path}")
    else:
        raw_path = RAW / f"trades_{stamp}.jsonl"
        print("collecting the trade stream")
        coins = resolve_coins(args.coins)
        n = asyncio.run(collect(coins, args.seconds, raw_path))
        print(f"  {n:,} trades -> {raw_path}")
        if n == 0:
            sys.exit("No trades received; check that api.hyperliquid.xyz is reachable")

    stream = analyze_stream(raw_path)
    wallets = [w for w, _ in stream["twap_wallet_ranking"][: args.wallets]]
    if not wallets:
        print("no TWAP wallets in the stream")
        labeled = {k: 0 for k in ("wallets_queried", "wallets_with_data", "metaorders",
                                  "slices_total", "slices_median", "slices_p90",
                                  "notional_median_usd", "notional_p10_usd",
                                  "notional_p90_usd", "duration_median_s", "coins_covered")}
        labeled["sample"] = []
        labeled["convention_check"] = {}
    else:
        print(f"userTwapSliceFills for {len(wallets)} wallets")
        labeled = fetch_labeled_metaorders(
            wallets, RAW / f"twapfills_{stamp}.json",
            aggressor_set=set(stream.get("aggressor_set") or []))

    v = verdict(stream, labeled)
    report(stream, labeled, v, args.seconds if not args.analyze_only else -1)

    out = ROOT / "data" / "feasibility_report.json"
    stream_slim = {k: val for k, val in stream.items()
                   if k not in ("twap_wallet_ranking", "aggressor_set")}
    stream_slim["twap_wallet_top20"] = stream["twap_wallet_ranking"][:20]
    out.write_text(json.dumps({
        "generated_utc": stamp,
        "params": vars(args),
        "stream": stream_slim,
        "labeled": labeled,
        "verdict": v,
    }, indent=2))
    print(f"wrote {out}")
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
