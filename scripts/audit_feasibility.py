#!/usr/bin/env python3
"""Follow-up audit of the feasibility check (check_feasibility.py).

Checks that zero-hash trades are TWAP slices (the TWAP share relies on it), that the ByTime
endpoints page past the 2000-item cap, how much the metaorders cut off by that cap change the
stats, and how random candidates compare with the top 25 the first check sampled. Results go
to data/feasibility_audit.json.

Reads the newest trades/twapfills pair in data/raw/ unless given paths.
    python3 scripts/audit_feasibility.py
    python3 scripts/audit_feasibility.py --control 12 --random 12
    python3 scripts/audit_feasibility.py --offline      # purity and truncation only, no network
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"

from glassbook.hl.stream import ZERO_HASH_PREFIX as ZERO_PREFIX, analyze_stream, post_info


def newest(pattern):
    files = sorted(RAW.glob(pattern))
    if not files:
        sys.exit(f"No {pattern} in data/raw/; run check_feasibility.py first")
    return files[-1]


def load_stream(path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def group_by_twap(fills_raw):
    g = defaultdict(list)
    for item in fills_raw:
        if not isinstance(item, dict):
            continue
        fill = item.get("fill", item)
        tid = item.get("twapId", fill.get("twapId"))
        if tid is not None:
            g[int(tid)].append(fill)
    return dict(g)


def tids_of(fills_raw):
    out = set()
    for item in fills_raw:
        if not isinstance(item, dict):
            continue
        fill = item.get("fill", item)
        if fill.get("tid") is not None:
            out.add(int(fill["tid"]))
    return out


def audit_zero_hash_purity(trades, twapfills):
    """Look up each zero-hash trade's tid among the sampled wallet's TWAP slices.

    The side-inferred taker should hit; the flipped side is the control.
    """
    sampled = {w.lower(): tids_of(f) for w, f in twapfills.items() if f}
    if not sampled:
        return {"error": "no usable TWAP slice sample"}

    hit_agg = miss_agg = hit_flip = miss_flip = 0
    checked = 0
    nonzero_hit = nonzero_checked = 0

    for t in trades:
        users = [u.lower() for u in (t.get("users") or []) if isinstance(u, str)]
        if len(users) < 2 or t.get("tid") is None:
            continue
        tid = int(t["tid"])
        agg = users[0] if t.get("side") == "B" else users[1]
        oth = users[1] if t.get("side") == "B" else users[0]
        zero = (t.get("hash") or "").startswith(ZERO_PREFIX)

        if zero:
            if agg in sampled:
                checked += 1
                if tid in sampled[agg]:
                    hit_agg += 1
                else:
                    miss_agg += 1
            if oth in sampled:
                if tid in sampled[oth]:
                    hit_flip += 1
                else:
                    miss_flip += 1
        else:
            # Reverse control: non-zero-hash trades should never be TWAP slices
            for u in users:
                if u in sampled:
                    nonzero_checked += 1
                    if tid in sampled[u]:
                        nonzero_hit += 1

    def r(a, b):
        return round(a / (a + b), 4) if (a + b) else None

    return {
        "zero_hash_checked": checked,
        "purity_aggressor": r(hit_agg, miss_agg),
        "purity_flipped": r(hit_flip, miss_flip),
        "n_aggressor": hit_agg + miss_agg,
        "n_flipped": hit_flip + miss_flip,
        "nonzero_false_positive_rate": r(nonzero_hit, nonzero_checked - nonzero_hit),
        "nonzero_checked": nonzero_checked,
    }


def purity_verdict(pa, pf):
    if pa is None and pf is None:
        return "nothing to compare (no sampled wallet in a zero-hash trade)"
    if pa is None or (pf is not None and pf > pa + 0.10):
        return "users order looks reversed"
    if pa >= 0.90:
        return "ok"
    if pa >= 0.60:
        # Liquidations / ADL / settlement?
        return "partly non-TWAP, label by tid rather than hash"
    return "zero hash doesn't identify TWAP fills"


def audit_pagination(wallet):
    probes = [
        ("userTwapSliceFillsByTime", {"type": "userTwapSliceFillsByTime", "user": wallet,
                                      "startTime": 0}),
        ("userFillsByTime", {"type": "userFillsByTime", "user": wallet, "startTime": 0}),
    ]
    out = {}
    for name, payload in probes:
        try:
            resp = post_info(payload, retries=1)
            out[name] = {"ok": isinstance(resp, list), "n": len(resp) if isinstance(resp, list) else 0}
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)[:120]}
        time.sleep(1.1)
    return out


def audit_truncation(twapfills):
    """Slice stats with and without metaorders cut off by the 2000-item cap."""
    all_slices, kept_slices = [], []
    capped_wallets = 0
    dropped = 0

    for wallet, raw in twapfills.items():
        if not raw:
            continue
        groups = group_by_twap(raw)
        if not groups:
            continue
        capped = len(raw) >= 2000
        capped_wallets += capped

        # In a capped window the metaorder owning the oldest fill is cut off
        times = [(int(f.get("time", 0)), tid) for tid, fs in groups.items() for f in fs]
        oldest_tid = min(times)[1] if times else None

        for tid, fs in groups.items():
            n = len(fs)
            all_slices.append(n)
            if capped and tid == oldest_tid:
                dropped += 1
                continue
            kept_slices.append(n)

    def stats(xs):
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        return {
            "n": len(s),
            "median": statistics.median(s),
            "mean": round(statistics.mean(s), 1),
            "p90": s[min(len(s) - 1, int(.9 * len(s)))],
            "max": s[-1],
        }

    return {
        "wallets_capped": capped_wallets,
        "wallets_total": sum(1 for v in twapfills.values() if v),
        "metaorders_dropped_at_boundary": dropped,
        "with_boundary": stats(all_slices),
        "boundary_excluded": stats(kept_slices),
    }


def audit_selection(candidates, sampled, n, label):
    """Metaorder sizes of n random unsampled candidates, to compare with the top N."""
    pool = [w for w in candidates if w.lower() not in sampled]
    if not pool:
        return {"label": label, "error": "empty candidate pool"}
    random.seed(20260823)
    pick = random.sample(pool, min(n, len(pool)))
    print(f"    Sampling {len(pick)} {label} (~1 req/s)")

    sizes, metaorders, with_data = [], 0, 0
    for i, w in enumerate(pick, 1):
        try:
            raw = post_info({"type": "userTwapSliceFills", "user": w}, retries=1)
        except Exception:
            time.sleep(1.1)
            continue
        raw = raw if isinstance(raw, list) else []
        if raw:
            with_data += 1
        g = group_by_twap(raw)
        metaorders += len(g)
        sizes.extend(len(v) for v in g.values())
        print(f"      [{i}/{len(pick)}] {w[:10]}... {len(raw):>5} slices, {len(g):>4} metaorders", flush=True)
        time.sleep(1.1)

    return {
        "label": label,
        "queried": len(pick),
        "with_data": with_data,
        "hit_rate": round(with_data / len(pick), 3) if pick else None,
        "metaorders": metaorders,
        "slices_median": statistics.median(sizes) if sizes else 0,
        "slices_mean": round(statistics.mean(sizes), 1) if sizes else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Follow-up audit of the feasibility check")
    ap.add_argument("--trades", type=str, default=None)
    ap.add_argument("--twapfills", type=str, default=None)
    ap.add_argument("--control", type=int, default=12, help="size of the non-aggressor control group")
    ap.add_argument("--random", type=int, default=12, help="size of the random-candidate control group")
    ap.add_argument("--offline", action="store_true", help="run only the offline checks (zero-hash purity, truncation)")
    args = ap.parse_args()

    tp = Path(args.trades) if args.trades else newest("trades_*.jsonl")
    fp = Path(args.twapfills) if args.twapfills else newest("twapfills_*.json")
    print(f"Stream {tp.name}\nTWAP   {fp.name}\n")

    trades = load_stream(tp)
    twapfills = json.loads(fp.read_text())
    stream = analyze_stream(tp)

    print("Zero-hash purity")
    purity = audit_zero_hash_purity(trades, twapfills)
    if "error" in purity:
        print(f"  {purity['error']}")
    else:
        pa, pf = purity["purity_aggressor"], purity["purity_flipped"]
        print(f"  Zero-hash trades with a sampled aggressor  {purity['n_aggressor']:,}")
        print(f"    tid found in its TWAP slices             {pa if pa is None else f'{pa:.1%}'}")
        print(f"  Control (direction flipped)                {purity['n_flipped']:,}"
              f"  hit {pf if pf is None else f'{pf:.1%}'}")
        fpr = purity["nonzero_false_positive_rate"]
        print(f"  Reverse control: non-zero hash in TWAP     "
              f"{fpr if fpr is None else f'{fpr:.2%}'} ({purity['nonzero_checked']:,} checked)")
        print()
        print("  purity: " + purity_verdict(pa, pf))

    print("\nTruncation at the 2000-item cap")
    trunc = audit_truncation(twapfills)
    print(f"  Wallets at the 2000-item cap      {trunc['wallets_capped']}/{trunc['wallets_total']}")
    print(f"  Truncated metaorders dropped      {trunc['metaorders_dropped_at_boundary']}")
    wb, be = trunc["with_boundary"], trunc["boundary_excluded"]
    print(f"\n  {'':<14}{'n_meta':>8}{'med_slc':>10}{'mean_slc':>10}{'p90':>8}{'max':>8}")
    print(f"  {'with boundary':<14}{wb.get('n',0):>8}{wb.get('median',0):>10.0f}"
          f"{wb.get('mean',0):>10.1f}{wb.get('p90',0):>8}{wb.get('max',0):>8}")
    print(f"  {'excl boundary':<14}{be.get('n',0):>8}{be.get('median',0):>10.0f}"
          f"{be.get('mean',0):>10.1f}{be.get('p90',0):>8}{be.get('max',0):>8}")
    if wb.get("n") and be.get("n"):
        d = (be["median"] - wb["median"]) / wb["median"] if wb["median"] else 0
        print(f"\n  Excluding boundary metaorders changes the median by {d:+.1%}.")

    out = {"trades_file": tp.name, "twapfills_file": fp.name, "zero_hash_purity": purity, "truncation": trunc}

    if not args.offline:
        print(f"\nPagination probe")
        probe_wallet = next(iter(twapfills))
        pag = audit_pagination(probe_wallet)
        for name, r in pag.items():
            if r.get("ok"):
                print(f"  ok   {name:<28} available, returned {r['n']:,} items")
            else:
                print(f"  FAIL {name:<28} {r.get('error','unavailable')}")
        out["pagination"] = pag

        print("\nSample selection bias")
        sampled = {w.lower() for w in twapfills}
        ranking = stream["twap_wallet_ranking"]
        aggr = set(stream.get("aggressor_set") or [])
        non_aggr = [w for w, _ in ranking if w not in aggr]
        all_cand = [w for w, _ in ranking]

        ctrl = audit_selection(non_aggr, sampled, args.control, "non-aggressor candidates")
        rnd = audit_selection(all_cand, sampled, args.random, "random candidates")

        print(f"\n  {'group':<26}{'hit rate':>9}{'metas':>8}{'med_slc':>10}{'mean_slc':>10}")
        # Reference row, hard-coded from check_feasibility.py's top-25 sample
        print(f"  {'top-25 (first sample)':<26}{'100.0%':>9}{243:>8}{141:>10}{'-':>10}")
        for r in (ctrl, rnd):
            if "error" in r:
                print(f"  {r['label']:<24}{r['error']}")
                continue
            hr_ = f"{r['hit_rate']:.1%}" if r["hit_rate"] is not None else "n/a"
            print(f"  {r['label']:<24}{hr_:>9}{r['metaorders']:>8}"
                  f"{r['slices_median']:>10.0f}{r['slices_mean']:>10.1f}")
        print()
        hr = ctrl.get("hit_rate")
        if hr is not None and hr > 0.3:
            # The wallet pool in collect_metaorders.py keeps inferred aggressors only
            print(f"  warning: {hr:.0%} of non-aggressor candidates have TWAP fills; an aggressor-only pool "
                  "misses them.")
        if rnd.get("slices_median") and rnd["slices_median"] < 141 * 0.5:
            print(f"  warning: random candidates have a median of {rnd['slices_median']:.0f} slices per "
                  "metaorder, vs 141 for the top 25; the top-25 sample overstates metaorder size.")
        out["selection"] = {"control": ctrl, "random": rnd}

    dst = ROOT / "data" / "feasibility_audit.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
