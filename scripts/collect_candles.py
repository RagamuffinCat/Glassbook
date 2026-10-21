#!/usr/bin/env python3
"""
Collect price candles (`candleSnapshot`) for the impact analysis. Each metaorder needs
[t_start - 24h, t_end + 2h] at the finest resolution that covers it. The assignment goes to
data/candles_plan.json, the bars to data/candles/.

The server keeps about 5000 bars per resolution, which for 1m is a bit over 3 days, so how far
back each one goes is probed first (on BTC).

Usage:
    python3 scripts/collect_candles.py --estimate     # offline: request count from the existing retention file (else assume 5000 bars)
    python3 scripts/collect_candles.py --probe        # network: probe retention only
    python3 scripts/collect_candles.py                # network: probe if needed, then plan and collect
    python3 scripts/collect_candles.py --resume
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"
CDIR = OUT / "candles"
RET_FILE = OUT / "candles_retention.json"
PLAN_FILE = OUT / "candles_plan.json"

from glassbook.candles import INTERVALS, MIN_MS, candle_path, choose_interval
from glassbook.hl.client import APIRequestFailed, default_client

POST_MS = 120 * MIN_MS          # Permanent impact uses +30 min, the rest is margin
VOL_MS = 24 * 60 * MIN_MS       # V is the traded volume over the 24h before the start
BARS_PER_REQ = 5000
BREAKER = 3
PROBE_COIN = "BTC"
PROBE_SPAN_MS = 400 * 1440 * MIN_MS


def merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def plan_requests(metas, retention):
    """Return (plan {(coin, interval): [(s, e), ...]}, assignment {(wallet, twap_id): interval | None})."""
    by = defaultdict(list)
    assign = {}
    for m in metas:
        s, e = int(m["t_start_ms"]), int(m["t_end_ms"])
        need_s, need_e = s - VOL_MS, e + POST_MS
        iv = choose_interval(need_s, retention)
        assign[(m["wallet"], m["twap_id"])] = iv
        if iv:
            by[(m["coin"], iv)].append((need_s, need_e))
    bar = dict(INTERVALS)
    plan = {}
    for (coin, iv), spans in by.items():
        b = bar[iv]
        chunks = []
        for s, e in merge_intervals(spans):
            s = s - s % b
            while s < e:
                chunks.append((s, min(e, s + BARS_PER_REQ * b)))
                s += BARS_PER_REQ * b
        plan[(coin, iv)] = chunks
    return plan, assign


def done_keys(plan):
    """Keys with a non-empty file on disk. After a crash the report can be behind the files."""
    return {k for k in plan if candle_path(CDIR, *k).exists() and candle_path(CDIR, *k).stat().st_size > 0}


def fetch(client, coin, interval, s, e):
    r = client.post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                      "startTime": int(s), "endTime": int(e)}})
    if not isinstance(r, list):
        raise APIRequestFailed(f"non-list response: {str(r)[:100]}")
    return r


def probe_retention(client, now_ms):
    """Fetch [now - 400 days, now] once per interval and record the earliest bar and bar count."""
    # TODO probe per coin? BTC is just a proxy, a coin listed later only shows up as low coverage
    out = {"probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "coin": PROBE_COIN,
           "earliest_ms": {}, "n_bars": {}}
    for name, bar in INTERVALS:
        rows = fetch(client, PROBE_COIN, name, now_ms - PROBE_SPAN_MS, now_ms)
        ts = [int(c.get("t", 0)) for c in rows if c.get("t")]
        out["earliest_ms"][name] = min(ts) if ts else None
        out["n_bars"][name] = len(ts)
        days = (now_ms - min(ts)) / (1440 * MIN_MS) if ts else None
        first = datetime.fromtimestamp(min(ts) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "n/a"
        print(f"  {name:<4} earliest {first} ({days and round(days, 1)} days ago)  {len(ts)} bars")
    return out


def main():
    ap = argparse.ArgumentParser(description="Collect price candles for the impact analysis (multiple resolutions)")
    ap.add_argument("--estimate", action="store_true", help="offline estimate, no requests")
    ap.add_argument("--probe", action="store_true", help="probe retention only (network, at most 6 requests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    run_id = json.loads((OUT / "collect_report.json").read_text())["run_id"]
    metas = [json.loads(l) for l in (OUT / "metaorders.jsonl").open() if l.strip()]
    now_ms = int(time.time() * 1000)

    client = None
    if not args.estimate:
        client = default_client()
        client.timeout = args.timeout
        print(f"API ok ({client.preflight()} s)")

    if args.estimate and not RET_FILE.exists():
        retention = {name: now_ms - BARS_PER_REQ * bar for name, bar in INTERVALS}
        print("(no retention file; estimating with 5000 bars per resolution)")
    elif args.probe or not RET_FILE.exists():
        print(f"\nprobing retention on {PROBE_COIN}")
        ret = probe_retention(client, now_ms)
        RET_FILE.write_text(json.dumps(ret, indent=2))
        retention = ret["earliest_ms"]
        if args.probe:
            print(f"\n  wrote {RET_FILE}")
            return 0
    else:
        ret = json.loads(RET_FILE.read_text())
        retention = ret["earliest_ms"]
        print(f"retention from {RET_FILE.name} ({ret['probed_at']})")

    plan, assign = plan_requests(metas, retention)
    n_req = sum(len(v) for v in plan.values())
    by_iv = defaultdict(int)
    for iv in assign.values():
        by_iv[iv or "none"] += 1
    print(f"\nrun {run_id}, metaorders {len(metas):,}, (coin, interval) pairs {len(plan)}, requests {n_req:,}, "
          f"worst-case weight {n_req * 120:,} (about {n_req * 120 / 900:.0f} min)")
    print(f"  metaorders per interval: {dict(sorted(by_iv.items()))} (none = no resolution covers it)")
    PLAN_FILE.write_text(json.dumps({"run_id": run_id, "retention": retention,
                                     "assign": [{"wallet": w, "twap_id": t, "interval": iv} for (w, t), iv in assign.items()],
                                     "n_requests": n_req}, indent=2))
    if args.estimate:
        return 0

    CDIR.mkdir(parents=True, exist_ok=True)
    rp = OUT / "candles_report.json"
    old = json.loads(rp.read_text()) if (args.resume and rp.exists()) else {}
    done = done_keys(plan) if args.resume else set()
    detail = {k: v for k, v in (old.get("detail") or {}).items()} if args.resume else {}
    empties = {tuple(x) for x in old.get("empty", [])} if args.resume else set()
    if args.resume:
        print(f"resume: {len(done)} (coin, interval) files present, {len(empties)} known to be empty; skipping them")
    crun = old.get("candles_run") or ("CND-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    failed = list(old.get("failed", []))
    bar = dict(INTERVALS)

    # Rewritten after every key, so after a crash the report still matches the files. --resume
    # also needs to know which keys came back empty.
    def write_report(st):
        rp.write_text(json.dumps({"candles_run": crun, "run_id": run_id, "status": st,
                                  "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "retention": retention, "n_keys": len(plan),
                                  "done": sorted([list(k) for k in done]), "empty": sorted([list(k) for k in empties]),
                                  "failed": failed, "detail": detail,
                                  "client": client.stats}, indent=2))

    t0 = time.monotonic()
    streak = 0
    keys = sorted(plan)
    for i, key in enumerate(keys, 1):
        if key in done or key in empties:
            continue
        coin, iv = key
        rows = {}
        try:
            for s, e in plan[key]:
                for c in fetch(client, coin, iv, s, e):
                    t = int(c.get("t", 0))
                    if t:
                        rows[t] = {"t": t, "o": c.get("o"), "h": c.get("h"), "l": c.get("l"),
                                   "c": c.get("c"), "v": c.get("v"), "n": c.get("n")}
        except APIRequestFailed as e:
            streak += 1
            failed.append({"coin": coin, "interval": iv, "error": str(e)[:160]})
            print(f"  [{i}/{len(keys)}] {coin:<12}{iv:<4} FAIL ({str(e)[:80]})")
            write_report("running")
            if streak >= BREAKER:
                sys.exit(f"{BREAKER} failures in a row, stopping (--resume to continue)")
            continue
        streak = 0
        expected = sum((e - s) // bar[iv] for s, e in plan[key])
        dk = f"{coin}|{iv}"
        if not rows:
            empties.add(key)          # Retention says there should be bars, so record it as empty, not done
            detail[dk] = {"n_candles": 0, "expected_bars": expected, "coverage": 0.0, "empty": True}
            print(f"  [{i}/{len(keys)}] {coin:<12}{iv:<4} empty (expected {expected} bars)")
            write_report("running")
            continue
        dst = candle_path(CDIR, coin, iv)
        tmp = dst.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for t in sorted(rows):
                f.write(json.dumps({"coin": coin, "interval": iv, **rows[t], "candles_run": crun}) + "\n")
        os.replace(tmp, dst)
        done.add(key)
        detail[dk] = {"n_candles": len(rows), "expected_bars": expected,
                      "coverage": round(len(rows) / expected, 4) if expected else None}
        print(f"  [{i}/{len(keys)}] {coin:<12}{iv:<4} {len(rows):>6} bars / expected {expected:>6} ({detail[dk]['coverage']:.1%})"
              f", {(time.monotonic() - t0) / 60:.1f} min", flush=True)
        write_report("running")

    for key in done:
        dk = f"{key[0]}|{key[1]}"
        if dk not in detail:
            n = sum(1 for _ in candle_path(CDIR, *key).open())
            expected = sum((e - s) // bar[key[1]] for s, e in plan[key])
            detail[dk] = {"n_candles": n, "expected_bars": expected,
                          "coverage": round(n / expected, 4) if expected else None, "from_file": True}
    status = "ok" if not failed else "degraded"
    write_report(status)
    low = [k for k, d in detail.items() if d.get("coverage") is not None and d["coverage"] < 0.9]
    print(f"\ncollected {len(done)}/{len(keys)}, empty {len(empties)}, failed {len(failed)}"
          f", coverage <90%: {len(low)}{': ' + ','.join(low[:8]) if low else ''}")
    print(f"API {client.stats['calls']} calls, waited {client.stats['waited_s'] / 60:.1f} min")
    print(f"wrote data/candles/*.jsonl, data/candles_report.json, data/candles_plan.json ({crun})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
