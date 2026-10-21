#!/usr/bin/env python3
"""Power analysis: how many labeled metaorders (and days of collecting) to pin down the impact
exponent beta.

x = Q/V is resampled from the measurable metaorders, I = A * x^beta + Gaussian noise with the
observed SD of I. --sigma-scale below 1 stands in for fill-level prices. A is set so that the top x
quartile has mean impact --target-bp. Interval width is 3.92 * SD(beta_hat); width <= 0.4 separates
beta 0.3 from 0.7, width <= 0.2 separates 0.3 from 0.5. The day counts assume 350 pool wallets
snapshotted every 3 days (roughly the retained TWAP history), no duplicates, constant measurable share.
All optimistic: a weaker true signal or wallet clustering needs a larger n.

Same inputs as estimate_impact.py. Writes impact_power.json, appends one row to trials.jsonl.

    python3 scripts/impact_power.py
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent.parent / "data"

from glassbook.impact import Candles, load_metas_and_assign, nls_beta, truth_rows


def calibrate_A(xs, beta, target_top_quartile_bp):
    q = sorted(xs)
    top = q[3 * len(q) // 4:]
    mean_xb = sum(v ** beta for v in top) / len(top)
    return target_top_quartile_bp / 1e4 / mean_xb


def simulate(xs, sigma, beta, A, n, k, rng):
    # TODO draw whole wallets, like the bootstrap does
    ests = []
    for _ in range(k):
        x = [rng.choice(xs) for _ in range(n)]
        y = [A * v ** beta + rng.gauss(0, sigma) for v in x]
        b = nls_beta(x, y)["beta"]
        if b is not None:
            ests.append(b)
    m = sum(ests) / len(ests)
    sd = math.sqrt(sum((e - m) ** 2 for e in ests) / len(ests))
    return {"n": n, "k": len(ests), "beta_hat_mean": round(m, 3), "beta_hat_sd": round(sd, 3),
            "ci95_width": round(3.92 * sd, 3), "hit_lo_edge": round(sum(1 for e in ests if e <= 0.03) / len(ests), 2),
            "hit_hi_edge": round(sum(1 for e in ests if e >= 1.97) / len(ests), 2)}


def main():
    ap = argparse.ArgumentParser(description="power analysis for the impact exponent")
    ap.add_argument("--k", type=int, default=60, help="simulations per n")
    ap.add_argument("--ns", default="700,2000,5000,10000,20000,50000")
    ap.add_argument("--betas", default="0.3,0.5,0.7")
    ap.add_argument("--target-bp", type=float, default=35.0, help="mean impact (bp) of the top x quartile, used to calibrate A")
    ap.add_argument("--sigma-scale", default="1.0,0.25",
                    help="noise SD as multiples of the observed SD (1.0 = candle closes, 0.25 = fill-level prices)")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output directory (default data/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metas, run_id, assign, retention, flagged = load_metas_and_assign(OUT)
    c = Candles(OUT / "candles")
    rows, states = truth_rows(metas, assign, flagged, c, "random", "with", 3)
    xs = [r["x"] for r in rows]
    ys = [r["I_tmp"] for r in rows]
    n0 = len(rows)
    mean_y = sum(ys) / n0
    sigma_obs = math.sqrt(sum((v - mean_y) ** 2 for v in ys) / n0)
    top = sorted(zip(xs, ys))[3 * n0 // 4:]
    top_q_mean_bp = sum(y for _, y in top) / len(top) * 1e4
    wallets = len({r["wallet"] for r in rows})
    per_wallet = n0 / wallets
    print(f"run {run_id}, measurable metaorders {n0} ({wallets} wallets, about {per_wallet:.1f} per wallet per snapshot), "
          f"sigma_I = {sigma_obs*1e4:.0f} bp, observed top-x-quartile mean impact {top_q_mean_bp:+.1f} bp "
          f"(calibration target {args.target_bp} bp)")

    rng = random.Random(args.seed)
    ns = [int(v) for v in args.ns.split(",")]
    betas = [float(v) for v in args.betas.split(",")]
    scales = [float(v) for v in args.sigma_scale.split(",")]
    grid = []
    for sc in scales:
        sigma = sigma_obs * sc
        for beta in betas:
            A = calibrate_A(xs, beta, args.target_bp)
            print(f"\n  noise sigma={sigma*1e4:.0f} bp (x{sc}), true beta={beta}, A={A:.3g}")
            print(f"  {'n':>7}{'b_hat mean':>11}{'b_hat SD':>9}{'width':>8}{'lo_edge':>8}{'hi_edge':>8}")
            for n in ns:
                k = args.k if n <= 10000 else max(20, args.k // 3)
                r = simulate(xs, sigma, beta, A, n, k, rng)
                r.update({"sigma_scale": sc, "sigma_bp": round(sigma * 1e4, 1), "beta_true": beta, "A": A})
                grid.append(r)
                print(f"  {n:>7}{r['beta_hat_mean']:>11.3f}{r['beta_hat_sd']:>9.3f}{r['ci95_width']:>8.3f}"
                      f"{r['hit_lo_edge']:>8.0%}{r['hit_hi_edge']:>8.0%}")

    # per (sigma, beta): smallest n on the grid that reaches each width; ">max_n" if none does
    thresholds = []
    for sc in scales:
        for beta in betas:
            rs = [r for r in grid if r["sigma_scale"] == sc and r["beta_true"] == beta]
            need = {}
            for w in (0.4, 0.2):
                ok = [r["n"] for r in rs if r["ci95_width"] <= w]
                need[f"width<={w}"] = min(ok) if ok else f">{max(ns)}"
            thresholds.append({"sigma_scale": sc, "beta_true": beta, **need})
    snap_per = per_wallet * 350
    print(f"\n  Minimum n per CI width, and collection days at about {snap_per:.0f} metaorders per snapshot"
          " (350 wallets, one snapshot every 3 days, no duplicates, constant measurable share):")

    def days(v):
        return "n/a" if isinstance(v, str) else f"{math.ceil(v / snap_per) * 3} days"
    for t in thresholds:
        print(f"    sigma x{t['sigma_scale']} beta={t['beta_true']}: "
              f"width<=0.4 needs n={t['width<=0.4']} ({days(t['width<=0.4'])}); "
              f"width<=0.2 needs n={t['width<=0.2']} ({days(t['width<=0.2'])})")
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": run_id,
           "observed": {"n": n0, "wallets": wallets, "per_wallet_per_snapshot": round(per_wallet, 2),
                        "sigma_I_bp": round(sigma_obs * 1e4, 1), "top_quartile_mean_bp": round(top_q_mean_bp, 1)},
           "assumptions": {"target_top_quartile_bp": args.target_bp, "pool_wallets": 350, "snapshot_every_days": 3,
                           "cluster_structure": "ignored (wallet clustering would raise the required n)",
                           "signal": "I = A * x^beta, A calibrated to the top-quartile mean impact; a weaker true signal would need a larger n"},
           "params": {k: v for k, v in vars(args).items() if k != "out"}, "grid": grid, "thresholds": thresholds,
           "metaorders_per_snapshot_350": round(snap_per)}
    dst = args.out / "impact_power.json"
    dst.write_text(json.dumps(out, indent=2))
    with (args.out / "trials.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": out["generated"], "script": "impact_power", "run_id": run_id, "kind": "power",
                            "thresholds": thresholds}) + "\n")
    print(f"\nwrote {dst}; appended 1 row to trials.jsonl\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
