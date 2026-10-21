#!/usr/bin/env python3
"""
Calibration checks for the paired-metaorder tests in detect_crossing.py.

The block-flip test is rerun with an independent seed (reseed). The false-positive check draws
K signal-free datasets from the block-flip null and reports the share with p <= 0.05, which
should be near 5%. The block-flip null takes the size/time coincidences as given, so the
coincidence tests check those directly by permuting start time, size or both within each coin.

Pass needs reseed p <= 0.05, false-positive rate <= --fpr-max and every coincidence test
p <= 0.05, in both the targeted and cross groups. A zero-variance null counts as a failure.

    python3 scripts/calibrate_pair_tests.py --reps 1000 --k 100 --seed 777
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"

from glassbook.pairs import (FLIP, calibration_problems, null_burst_flip, offsetting_pairs,
                             perm_dist, read_jsonl, record_trial, split_bursts)

VARIANTS = ("time", "size", "time_size")


def burst_flip_dataset(metas, start_tol_s, rng):
    """One signal-free dataset drawn from the block-flip null."""
    out = []
    for b in split_bursts(metas, start_tol_s):
        flip = rng.random() < 0.5
        for m in b:
            sd = m.get("side")
            out.append({**m, "side": FLIP.get(sd, sd) if flip else sd})
    return out


def block_flip_p(metas, size_tol, start_tol_s, pair_filter, reps, seed):
    obs = sum(1 for p in offsetting_pairs(metas, size_tol, start_tol_s)["pairs"] if pair_filter(p))
    counts = null_burst_flip(metas, size_tol, start_tol_s, pair_filter, reps=reps, seed=seed)
    return {"observed": obs, **perm_dist(counts, obs)}


def fpr(metas, size_tol, start_tol_s, pair_filter, k, reps, seed):
    """Block-flip p-value distribution over k datasets drawn from the block-flip null."""
    rng = random.Random(seed)
    ps = []
    for i in range(k):
        synth = burst_flip_dataset(metas, start_tol_s, rng)
        r = block_flip_p(synth, size_tol, start_tol_s, pair_filter, reps, seed=seed + 1000 + i)
        ps.append(r["p_value"])
    ps.sort()
    return {"k": k, "reps_each": reps, "fpr_005": round(sum(p <= 0.05 for p in ps) / k, 4),
            "fpr_010": round(sum(p <= 0.10 for p in ps) / k, 4), "p_min": ps[0], "p_median": ps[k // 2]}


def coincidence_null(metas, size_tol, start_tol_s, pair_filter, variant, reps, seed):
    """Permute t_start, size or both within each coin. Counts opposite-side matches passing pair_filter."""
    rng = random.Random(seed)
    obs = sum(1 for p in offsetting_pairs(metas, size_tol, start_tol_s)["pairs"] if pair_filter(p))
    by_coin = defaultdict(list)
    for m in metas:
        by_coin[m.get("coin")].append(m)
    counts = []
    for _ in range(reps):
        shuffled = []
        for rows in by_coin.values():
            ts = [r["t_start_ms"] for r in rows]
            sz = [r["size"] for r in rows]
            if variant in ("time", "time_size"):
                rng.shuffle(ts)
            if variant in ("size", "time_size"):
                rng.shuffle(sz)
            shuffled += [{**r, "t_start_ms": t, "size": s} for r, t, s in zip(rows, ts, sz)]
        counts.append(sum(1 for p in offsetting_pairs(shuffled, size_tol, start_tol_s)["pairs"]
                          if pair_filter(p)))
    return {"variant": variant, "observed": obs, **perm_dist(counts, obs)}


def main():
    ap = argparse.ArgumentParser(description="calibration checks for the paired-metaorder tests")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100, help="number of null datasets for the false-positive-rate check")
    ap.add_argument("--k-reps", type=int, default=200, help="permutations per null dataset")
    ap.add_argument("--seed", type=int, default=777, help="independent seed (not detect_crossing's 20260923)")
    ap.add_argument("--size-tol", type=float, default=0.05)
    ap.add_argument("--start-tol", type=float, default=60)
    ap.add_argument("--fpr-max", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output dir, default data/")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trials = args.out / "trials.jsonl"

    try:
        metas = read_jsonl(OUT / "metaorders.jsonl")
    except FileNotFoundError as e:
        sys.exit(str(e))
    run_id = json.loads((OUT / "collect_report.json").read_text())["run_id"]
    tgt = [m for m in metas if m.get("sampling") == "targeted"]
    not_self = lambda p: not p["same_wallet"]
    cross_f = lambda p: p["cross_layer"]
    st, sz = args.start_tol, args.size_tol
    params = dict(vars(args))
    del params["out"]
    res = {"run_id": run_id, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "params": params}

    print(f"run {run_id}, independent seed {args.seed}")
    print("\nBlock-flip test rerun with an independent seed")
    res["reseed_targeted"] = block_flip_p(tgt, sz, st, not_self, args.reps, args.seed)
    res["reseed_cross"] = block_flip_p(metas, sz, st, cross_f, args.reps, args.seed + 1)
    for k in ("reseed_targeted", "reseed_cross"):
        r = res[k]
        print(f"  {k:<16} observed {r['observed']}, null {r['mean']} (p95 {r['p95']}, max {r['max']}), "
              f"p={r['p_value']}" + ("  [null has zero variance]" if r["degenerate"] else ""))

    print(f"\nFalse-positive rate: {args.k} signal-free datasets, {args.k_reps} permutations each")
    res["false_positive_targeted"] = fpr(tgt, sz, st, not_self, args.k, args.k_reps, args.seed + 10)
    res["false_positive_cross"] = fpr(metas, sz, st, cross_f, max(20, args.k // 2), args.k_reps, args.seed + 20)
    for k in ("false_positive_targeted", "false_positive_cross"):
        r = res[k]
        print(f"  {k:<24} FPR(0.05)={r['fpr_005']:.3f}, FPR(0.10)={r['fpr_010']:.3f}, "
              f"p min {r['p_min']}, median {r['p_median']} (k={r['k']})")

    print("\nSize/time coincidence tests (start time, size or both permuted within each coin)")
    res["coincidence"] = {}
    for layer, rows, flt in (("targeted", tgt, not_self), ("cross", metas, cross_f)):
        for v in VARIANTS:
            r = coincidence_null(rows, sz, st, flt, v, args.reps, args.seed + 100 + len(res["coincidence"]))
            res["coincidence"][f"{layer}_{v}"] = r
            print(f"  {layer:<9}{v:<10} observed {r['observed']}, null {r['mean']} (p95 {r['p95']}, max {r['max']}), "
                  f"p={r['p_value']}" + ("  [null has zero variance]" if r["degenerate"] else ""))

    problems = calibration_problems(res, args.fpr_max)
    res["verdict"] = "pass" if not problems else "fail"
    res["problems"] = problems
    # detect_crossing.py reads the verdict and reports "calibration_failed" if it's fail
    print(f"\nverdict {res['verdict']}" + "".join("\n  - " + x for x in problems))

    dst = args.out / "pair_test_calibration.json"
    dst.write_text(json.dumps(res, indent=2))
    base = {"script": "calibrate_pair_tests", "run_id": run_id,
            "params": params}
    for k in ("reseed_targeted", "reseed_cross"):
        record_trial(trials, {**base, "test": k, "p_value": res[k]["p_value"]})
    for k in ("false_positive_targeted", "false_positive_cross"):
        record_trial(trials, {**base, "test": k, "fpr_005": res[k]["fpr_005"]})
    for k, r in res["coincidence"].items():
        record_trial(trials, {**base, "test": f"coincidence_{k}", "p_value": r["p_value"]})
    print(f"\nwrote {dst}; appended {2 + 2 + len(res['coincidence'])} rows to trials.jsonl\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
