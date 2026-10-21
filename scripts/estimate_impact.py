#!/usr/bin/env python3
"""Impact law on candles: fit I = A * x^beta for the labeled metaorders and for pseudo-metaorders
rebuilt by the heuristics on the mixed stream. The beta gap between the two is the definition bias.

Reads metaorders.jsonl, candles_plan.json, crossing_report.json and candles/ from data/. Writes
impact_report.json and appends one row per estimate to trials.jsonl.

    python3 scripts/estimate_impact.py
"""

import argparse
import json
import math
import random
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
OUT = ROOT / "data"
CDIR = OUT / "candles"

from glassbook.candles import choose_interval
from glassbook.impact import (TIERS, VOL_MS, Candles, impact_row, load_metas_and_assign,
                              nls_beta, truth_rows)
from glassbook.metaorder.dataset import focus_for, load_dataset
from glassbook.metaorder.heuristics import h1_gap_runs, h3_direction_runs, sort_fills


def ols(lx: list[float], ly: list[float]) -> tuple[float, float] | None:
    n = len(lx)
    if n < 3:
        return None
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx <= 0:
        return None
    b = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / sxx
    return my - b * mx, b


def binned_beta(x: list[float], y: list[float], nbins: int = 10) -> dict:
    """Bin by quantiles of log x, take the median I per bin, fit log-log OLS on the bins with a positive median."""
    pairs = sorted(zip(x, y))
    n = len(pairs)
    if n < nbins * 5:
        return {"beta": None, "bins_used": 0, "why": "too few samples"}
    lx, ly = [], []
    for k in range(nbins):
        seg = pairs[k * n // nbins:(k + 1) * n // nbins]
        if not seg:
            continue
        ys = sorted(v for _, v in seg)
        med = ys[len(ys) // 2]
        xc = math.exp(sum(math.log(v) for v, _ in seg) / len(seg))
        if med > 0:
            lx.append(math.log(xc))
            ly.append(math.log(med))
    if len(lx) < 5:
        return {"beta": None, "bins_used": len(lx), "why": "fewer than 5 bins with a positive median"}
    r = ols(lx, ly)
    return {"beta": round(r[1], 4) if r else None, "bins_used": len(lx), "n_bins": nbins}


def ols_positive(x, y):
    keep = [(a, b) for a, b in zip(x, y) if b > 0]
    if len(keep) < 3:
        return {"beta": None, "dropped_frac": round(1 - len(keep) / max(1, len(x)), 4)}
    r = ols([math.log(a) for a, _ in keep], [math.log(b) for _, b in keep])
    return {"beta": round(r[1], 4) if r else None, "dropped_frac": round(1 - len(keep) / len(x), 4), "n_used": len(keep)}


def bootstrap_beta(rows, field, reps, seed=20260923):
    """Wallet-clustered bootstrap, percentile 95% CI of the NLS beta.

    One wallet's metaorders aren't independent of each other, so whole wallets are drawn. The
    draws follow the sorted wallet ids, which means the CI changes with the pseudonyms.
    """
    by_w = defaultdict(list)
    for r in rows:
        by_w[r["wallet"]].append(r)
    wallets = sorted(by_w)
    rng = random.Random(seed)
    bs = []
    for _ in range(reps):
        samp = [by_w[w] for w in (rng.choice(wallets) for _ in wallets)]
        xs = [r["x"] for grp in samp for r in grp]
        ys = [r[field] for grp in samp for r in grp]
        b = nls_beta(xs, ys)["beta"]
        if b is not None:
            bs.append(b)
    bs.sort()
    if len(bs) < 20:
        return {"reps": len(bs), "ci95": None}
    return {"reps": len(bs), "ci95": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]],
            "n_wallets": len(wallets)}


def estimate_group(rows, field, reps):
    """NLS beta (main) plus binned-median and positive-only OLS (reference) for one group.

    state is "untestable" if n < 100 or x spans less than one decade, "unidentified" if the 95% CI is missing
    or at least 1.0 wide, else "ok".
    """
    rows = [r for r in rows if r.get(field) is not None]
    n = len(rows)
    if n < 100:
        return {"state": "untestable", "why": f"n={n} < 100", "n": n}
    xs = [r["x"] for r in rows]
    span = math.log10(max(xs)) - math.log10(min(xs))
    if span < 1:
        return {"state": "untestable", "why": f"log10(Q/V) span {span:.2f} < 1", "n": n}
    ys = [r[field] for r in rows]
    nls = nls_beta(xs, ys)
    bs = bootstrap_beta(rows, field, reps)
    ci = bs.get("ci95")
    # point estimate alone means little, the bootstrap CI can span the whole search range
    state = "ok" if (ci and (ci[1] - ci[0]) < 1.0) else "unidentified"
    return {"state": state, "n": n, "n_wallets": len({r["wallet"] for r in rows}),
            "why": None if state == "ok" else f"95% CI width of beta {round(ci[1] - ci[0], 2) if ci else 'n/a'} >= 1.0",
            "x_log10_span": round(span, 2), "x_median": sorted(xs)[n // 2],
            "I_mean_bp": round(sum(ys) / n * 1e4, 2), "I_median_bp": round(sorted(ys)[n // 2] * 1e4, 2),
            "I_sd_bp": round(math.sqrt(sum((v - sum(ys) / n) ** 2 for v in ys) / n) * 1e4, 1),
            "I_pos_frac": round(sum(1 for v in ys if v > 0) / n, 3),
            "nls": nls, "nls_bootstrap": bs,
            "binned": binned_beta(xs, ys), "ols_positive": ols_positive(xs, ys)}


def pseudo_metas_from_clusters(fills, clustering):
    by = defaultdict(list)
    for f in fills:
        by[clustering[f.fid]].append(f)
    out = []
    for grp in by.values():
        grp.sort(key=lambda f: f.ts_ms)
        sides = {f.side for f in grp}
        if len(sides) == 1:
            side = grp[0].side
        else:
            # Doesn't happen with h1 and h3, they never mix sides. Here for other rules
            side = "B" if sum(f.sz for f in grp if f.side == "B") >= sum(f.sz for f in grp if f.side == "A") else "A"
        out.append({"wallet": grp[0].wallet, "coin": grp[0].coin, "side": side,
                    "t_start_ms": grp[0].ts_ms, "t_end_ms": grp[-1].ts_ms,
                    "size": sum(f.sz for f in grp), "n_fills": len(grp)})
    return out


def heuristic_rows(metas, flagged, retention, c, min_bars, tol_ms):
    """Impact rows for pseudo-metaorders rebuilt on the mixed stream (random layer, truncated dropped, pairs kept)."""
    ds = load_dataset(OUT, mixed=True, layers=("random",))      # Same sample as benchmark_reconstruction.py
    _, keep = focus_for(ds, "drop_truncated", "with", flagged)
    sub = sort_fills(f for f in ds.fills if ds.meta_key_of_fid.get(f.fid) is None or ds.meta_key_of_fid[f.fid] in keep)
    specs = [("h1@60", lambda fs: h1_gap_runs(fs, tau_s=60)), ("h1@300", lambda fs: h1_gap_runs(fs, tau_s=300)),
             ("h1@3600", lambda fs: h1_gap_runs(fs, tau_s=3600)), ("h3@300", lambda fs: h3_direction_runs(fs, tau_s=300))]
    out = {}
    for name, fn in specs:
        pm = pseudo_metas_from_clusters(sub, fn(sub))
        rows, states = [], defaultdict(int)
        for m in pm:
            iv = choose_interval(m["t_start_ms"] - VOL_MS, retention)
            if not iv:
                states["no_interval"] += 1
                continue
            r = impact_row(c, m, iv, min_bars)
            states[r["state"]] += 1
            if r["state"] == "ok":
                rows.append(r)
        out[name] = {"rows": rows, "states": dict(states), "n_pseudo": len(pm)}
    return out, {"n_fills": len(sub), "n_true_metaorders": len(keep)}


def main():
    ap = argparse.ArgumentParser(description="impact law: ground-truth vs heuristic metaorders")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap reps for main estimates")
    ap.add_argument("--reps-minor", type=int, default=200,
                    help="bootstrap reps for the targeted layer, sensitivity runs and subsamples")
    ap.add_argument("--no-heuristics", action="store_true")
    ap.add_argument("--tol", type=float, default=600)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write impact_report.json; trials.jsonl gets appended there as well (default: data/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metas, run_id, assign, retention, flagged = load_metas_and_assign(OUT)
    c = Candles(CDIR)
    impact_run = "IMP-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    print(f"run {impact_run}, data run {run_id}, metaorders {len(metas):,}, paired flags {len(flagged)}")
    results = {"truth": {}, "heuristics": {}}

    def report(tag, rows, states, reps, extra=None):
        blk = {"states": states, "n_ok": len(rows), **(extra or {})}
        for field in ("I_tmp", "I_perm"):
            blk[field] = {"pooled": estimate_group(rows, field, reps)}
            for iv in TIERS:
                sub = [r for r in rows if r["interval"] == iv]
                if sub:
                    blk[field][iv] = estimate_group(sub, field, max(50, reps // 5))
        return blk

    def show(tag, blk):
        for field in ("I_tmp", "I_perm"):
            p = blk[field]["pooled"]
            if p["state"] == "untestable":
                print(f"  {tag:<34}{field:<7} untestable: {p['why']}")
                continue
            if p["state"] == "unidentified":
                print(f"  {tag:<34}{field:<7} unidentified: {p['why']}; n={p['n']}, mean I {p['I_mean_bp']:+.1f} bp"
                      f" (SD {p['I_sd_bp']} bp), I>0 {p['I_pos_frac']:.0%}")
            ci = p["nls_bootstrap"].get("ci95")
            print(f"  {tag:<34}{field:<7} n={p['n']:>5} wallets {p['n_wallets']:>3}, beta_NLS {p['nls']['beta']:.3f}"
                  f" CI [{ci[0]:.3f},{ci[1]:.3f}]" if ci else f"  {tag:<34}{field:<7} n={p['n']:>5}, beta_NLS {p['nls']['beta']:.3f}, no CI",
                  end="")
            print(f", beta_binned {p['binned']['beta'] if p['binned']['beta'] is not None else 'n/a'}"
                  f", beta_OLS+ {p['ols_positive']['beta'] if p['ols_positive']['beta'] is not None else 'n/a'}"
                  f" (dropped {p['ols_positive']['dropped_frac']:.0%}), I>0 {p['I_pos_frac']:.0%}, R2 {p['nls']['r2']}")
            tiers = {iv: blk[field][iv] for iv in TIERS if iv in blk[field]}
            print("      per-interval beta_NLS: " + ", ".join(
                f"{iv} {t['nls']['beta']:.2f}(n={t['n']})" if t["state"] == "ok" else f"{iv} n/a (n={t['n']})"
                for iv, t in tiers.items()))

    print(f"\nGround-truth metaorders (truncated ones excluded)")
    for min_bars, reps in ((3, args.reps), (1, args.reps_minor)):
        print(f"\n  metaorder duration >= {min_bars} candles ({'main' if min_bars == 3 else 'sensitivity'})")
        for layer in ("random", "targeted"):
            for pairing in ("with", "without"):
                rows, states = truth_rows(metas, assign, flagged, c, layer, pairing, min_bars)
                tag = f"{layer}/{pairing}/min{min_bars}bars"
                blk = report(tag, rows, states, reps if layer == "random" else args.reps_minor)
                results["truth"][tag] = blk
                print(f"  [{tag}] measurable {len(rows)}, states {states}")
                show(tag, blk)

    print("\nLarge-metaorder subsamples (random layer, truncated dropped, with pairs; selected by size)")
    for min_bars in (3, 1):
        rows, states = truth_rows(metas, assign, flagged, c, "random", "with", min_bars)
        for name, flt in (("QV>=1e-3", lambda r: r["x"] >= 1e-3),
                          ("notional>=50k", lambda r: (r.get("notional") or 0) >= 50_000)):
            sub = [r for r in rows if flt(r)]
            tag = f"random/large[{name}]/min{min_bars}bars"
            blk = report(tag, sub, {"selected": len(sub), "of": len(rows)}, args.reps_minor)
            blk["caveat"] = "subsample selected by size after the fact; beta not identified, wide CI"
            results["truth"][tag] = blk
            print(f"  [{tag}] selected {len(sub)}/{len(rows)}")
            show(tag, blk)

    if not args.no_heuristics:
        print("\nHeuristic metaorders (mixed stream; random layer, truncated dropped, with pairs; duration >= 3 candles)")
        hr, hinfo = heuristic_rows(metas, flagged, retention, c, 3, int(args.tol * 1000))
        results["heuristics_info"] = hinfo
        truth_main = results["truth"]["random/with/min3bars"]
        for name, d in hr.items():
            blk = report(name, d["rows"], d["states"], args.reps, {"n_pseudo": d["n_pseudo"]})
            results["heuristics"][name] = blk
            print(f"  [{name}] pseudo-metaorders {d['n_pseudo']:,}, measurable {len(d['rows'])}, states {d['states']}")
            show(name, blk)
            for field in ("I_tmp", "I_perm"):
                a, b = truth_main[field]["pooled"], blk[field]["pooled"]
                if a["state"] == "ok" and b["state"] == "ok":
                    d_beta = round(b["nls"]["beta"] - a["nls"]["beta"], 4)
                    blk[field]["beta_minus_truth"] = d_beta
                    print(f"      {field} beta_heuristic - beta_truth = {d_beta:+.3f}")
                else:
                    blk[field]["beta_minus_truth"] = None
                    print(f"      {field} no beta difference: truth {a['state']}, heuristic {b['state']}")

    out = {"impact_run": impact_run, "data_run_id": run_id, "retention": retention,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "params": {k: v for k, v in vars(args).items() if k != "out"}, "results": results}
    dst = args.out / "impact_report.json"
    dst.write_text(json.dumps(out, indent=2))
    n = 0
    with (args.out / "trials.jsonl").open("a", encoding="utf-8") as f:
        for kind in ("truth", "heuristics"):
            for tag, blk in results[kind].items():
                for field in ("I_tmp", "I_perm"):
                    p = blk[field]["pooled"]
                    f.write(json.dumps({"ts": out["generated"], "script": "estimate_impact", "impact_run": impact_run,
                                        "run_id": run_id, "kind": kind, "group": tag, "field": field, "state": p["state"],
                                        "n": p.get("n"), "beta_nls": (p.get("nls") or {}).get("beta"),
                                        "ci95": (p.get("nls_bootstrap") or {}).get("ci95"),
                                        "beta_binned": (p.get("binned") or {}).get("beta"),
                                        "beta_ols_pos": (p.get("ols_positive") or {}).get("beta")}) + "\n")
                    n += 1
    print(f"\nwrote {dst}; appended {n} rows to trials.jsonl\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
