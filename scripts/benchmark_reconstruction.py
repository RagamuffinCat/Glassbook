#!/usr/bin/env python3
"""Reconstruction-error benchmark: score the metaorder heuristics against the twapId labels.

Writes out/reconstruction_<stream>.json (layers other than random get the layer in the name).
--log-trials also appends the estimates to <data>/trials.jsonl.

    python3 scripts/benchmark_reconstruction.py                      # TWAP-only and mixed stream, random layer
    python3 scripts/benchmark_reconstruction.py --mode mixed         # mixed stream only
    python3 scripts/benchmark_reconstruction.py --layers targeted    # targeted layer
    python3 scripts/benchmark_reconstruction.py --breakdown          # strata: overlap, size, coin
    python3 scripts/benchmark_reconstruction.py --data examples/sample
"""

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbook.metaorder.dataset import BOUNDARY_RULES, Dataset, focus_for, load_dataset, read_json
from glassbook.metaorder.heuristics import (HEURISTICS, economic_truth, evaluate,
                                            h3_direction_runs, is_noise_key, sort_fills, sweep)

TAUS = (60, 300, 900, 1800, 3600)
CELLS = [(b, p) for b in BOUNDARY_RULES for p in ("with", "without")]


def flagged_set(data_dir):
    """(wallet, twap_id) of paired metaorders. Without crossing_report.json nothing is flagged and "with" = "without"."""
    p = data_dir / "crossing_report.json"
    if not p.exists():
        return set()
    return {(x[k]["wallet"], x[k]["twap_id"]) for x in read_json(p)["all_pairs"] for k in ("a", "b")}


def econ_focus(ds, econ, keep):
    """Economic-truth focus: a merged group is scored if it holds any scored twapId metaorder."""
    members = defaultdict(set)
    for fid, k in econ.items():
        if not is_noise_key(k):
            members[k].add(ds.meta_key_of_fid[fid])
    scored = {k for k, ms in members.items() if ms & keep}
    return lambda tk: tk in scored


def run_cells(ds, flagged, min_fills):
    """Per (boundary, pairing) cell, sweep against twapId and economic truth. Mixed stream also gets the oid variant."""
    econ = economic_truth(ds.truth, ds.fills)
    out = []
    for boundary, pairing in CELLS:
        focus, keep = focus_for(ds, boundary, pairing, flagged)
        n_scored = sum(1 for f in ds.fills if ds.meta_key_of_fid.get(f.fid) in keep)
        tag = f"{boundary}/{pairing}"
        if n_scored < min_fills:
            out.append({"boundary": boundary, "pairing": pairing, "state": "untestable",
                        "why": f"scored fills {n_scored} < {min_fills}"})
            print(f"[{tag}] untestable: scored fills {n_scored}")
            continue
        # cluster-then-select: rules see the whole tape, the cell only picks what gets scored.
        # truncated / paired metaorders still interfere with the rest, as on real data
        rows = sweep(ds.truth, ds.fills, taus=TAUS, focus=focus)
        rows_econ = sweep(econ, ds.fills, taus=TAUS, focus=econ_focus(ds, econ, keep))
        cell = {"boundary": boundary, "pairing": pairing, "state": "ok", "n_scored_metaorders": len(keep),
                "n_scored_fills": n_scored, "n_fills_in_stream": len(ds.fills), "rows": rows, "rows_economic": rows_econ}
        if ds.mixed:
            cell["rows_oid"] = [{"heuristic": r["heuristic"], "tau_s": r["tau_s"],
                                 **{k: v for k, v in evaluate(ds.truth_oid, pred, fills=ds.fills, focus=focus).items()
                                    if k.startswith("pair_") or k == "jaccard_focus"}}
                                for r, pred in _preds(ds, rows)]
        out.append(cell)
        print(f"[{tag}] metaorders {len(keep):,}, scored fills {n_scored:,}, tape {len(ds.fills):,} fills")
        _print_rows(rows, rows_econ)
    return out


def _preds(ds, rows):
    fills = sort_fills(ds.fills)
    for r in rows:
        fn = HEURISTICS[r["heuristic"]]
        kw = {"window_s": r["tau_s"]} if r["heuristic"] == "h2_fixed_window" else {"tau_s": r["tau_s"]}
        yield r, fn(fills, **kw)


def _print_rows(rows, rows_econ=None):
    print(f"  {'heuristic':<18}{'tau':>5}{'F1all':>7}{'F1foc':>7}{'Jacc':>7}{'exact':>7}{'mrgTWAP':>8}{'contam':>7}"
          f"{'split':>7}{'ntl_med':>9}{'ntl_p90':>9}{'ntl_geo':>8}{'>2x':>6}{'>10x':>6}{'agg':>7}")
    for r in rows:
        print(f"  {r['heuristic']:<18}{r['tau_s']:>5.0f}{_f(r['pair_f1']):>7}{_f(r['pair_f1_focus']):>7}"
              f"{_f(r['jaccard_focus']):>7}"
              f"{_pct(r['exact_recovered_rate']):>7}{_pct(r['merge_twap_rate']):>8}{_pct(r['contam_nontwap_rate']):>7}"
              f"{_pct(r['split_rate']):>7}{_f(r['notional_ratio_median'], 2):>9}{_f(r['notional_ratio_p90'], 2):>9}"
              f"{_f(r['notional_ratio_geomean'], 2):>8}{_pct(r['notional_ratio_share_gt2']):>6}"
              f"{_pct(r['notional_ratio_share_gt10']):>6}{_f(r['notional_ratio_aggregate'], 2):>7}")
    if rows_econ:
        e = {(r["heuristic"], r["tau_s"]): r for r in rows_econ}
        print("  (economic-metaorder truth, h1) " + ", ".join(
            f"tau={t:.0f} F1foc {_f(e[('h1_gap_runs', t)]['pair_f1_focus'])}"
            f" mrgTWAP {_pct(e[('h1_gap_runs', t)]['merge_twap_rate'])}"
            for t in TAUS))


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def bl_rule(ds, flagged, tau_s=1800.0, min_fills=10, max_hours=24.0):
    """Barone-Lillo style rule: h3 on taker fills only, keeping clusters of at least min_fills fills within max_hours.

    Reports how many scored metaorders it covers (by count and notional) and the metrics on those.
    """
    # B&L count child orders, we count fills (a child can sweep several levels) -> min_fills=10 is easier here
    fills = sort_fills(ds.fills)
    active = [f for f in fills if f.crossed is True]
    n_unknown = sum(1 for f in fills if f.crossed is None)
    pred_raw = h3_direction_runs(active, tau_s=tau_s)
    by_c = defaultdict(list)
    for f in active:
        by_c[pred_raw[f.fid]].append(f)
    retained = {c for c, fs in by_c.items()
                if len(fs) >= min_fills and (max(x.ts_ms for x in fs) - min(x.ts_ms for x in fs)) <= max_hours * 3600_000}
    # dropped fills -> singletons, evaluate() needs every fid
    pred = {}
    for f in fills:
        c = pred_raw.get(f.fid)
        pred[f.fid] = ("bl", c) if c in retained else ("bl_single", f.fid)
    focus_all, keep = focus_for(ds, "drop_truncated", "with", flagged)
    covered = {ds.meta_key_of_fid[f.fid] for f in fills
               if ds.meta_key_of_fid.get(f.fid) in keep and pred[f.fid][0] == "bl"}
    notional_by_meta = defaultdict(float)
    for f in fills:
        k = ds.meta_key_of_fid.get(f.fid)
        if k in keep:
            notional_by_meta[k] += f.notional
    cov_n = sum(notional_by_meta[k] for k in covered)
    tot_n = sum(notional_by_meta.values())
    m = evaluate(ds.truth, pred, fills=fills,
                 focus=lambda tk: isinstance(tk, tuple) and tk[0] == "truth" and (tk[1], tk[2]) in covered)
    res = {"rule": f"crossed fills only, h3 direction runs tau={tau_s:.0f}s, "
                   f"clusters with >={min_fills} fills spanning <={max_hours:.0f}h",
           "n_active_fills": len(active), "n_fills_crossed_unknown": n_unknown, "n_clusters_raw": len(by_c),
           "n_clusters_retained": len(retained), "n_scored_metaorders": len(keep), "n_covered_metaorders": len(covered),
           "coverage_by_count": round(len(covered) / len(keep), 4) if keep else None,
           "coverage_by_notional": round(cov_n / tot_n, 4) if tot_n else None,
           "metrics_on_covered": {k: m[k] for k in ("exact_recovered_rate", "merge_rate", "merge_twap_rate", "contam_nontwap_rate",
                                                     "split_rate", "notional_ratio_median", "notional_ratio_p90",
                                                     "notional_ratio_geomean", "notional_ratio_share_gt2",
                                                     "notional_ratio_share_gt10", "notional_ratio_aggregate",
                                                     "pair_f1_focus", "jaccard_focus")}}
    print(f"\n[BL rule] taker fills {len(active):,} (crossed unknown {n_unknown}), clusters {len(by_c):,}, "
          f"retained {len(retained):,}, covered scored metaorders {len(covered)}/{len(keep)} "
          f"(by notional {_pct(res['coverage_by_notional'])})")
    mc = res["metrics_on_covered"]
    print(f"  covered metaorders: exact {_pct(mc['exact_recovered_rate'])}, merged TWAP {_pct(mc['merge_twap_rate'])}, "
          f"contam non-TWAP {_pct(mc['contam_nontwap_rate'])}, split {_pct(mc['split_rate'])}, "
          f"notional ratio median {_f(mc['notional_ratio_median'], 2)} p90 {_f(mc['notional_ratio_p90'], 2)} "
          f"geomean {_f(mc['notional_ratio_geomean'], 2)} >2x {_pct(mc['notional_ratio_share_gt2'])}, "
          f"F1 focus {_f(mc['pair_f1_focus'])}")
    return res


def overlap_flags(metas):
    by = {}
    for m in metas:
        by.setdefault((m["wallet"], m["coin"], m["side"]), []).append(m)
    out = {}
    for rows in by.values():
        rows.sort(key=lambda m: m["t_start_ms"])
        for i, a in enumerate(rows):
            ov = False
            for b in rows[i + 1:]:
                if b["t_start_ms"] > a["t_end_ms"]:
                    break
                ov = True
                out[(b["wallet"], b["twap_id"])] = True
            key = (a["wallet"], a["twap_id"])
            out[key] = out.get(key, False) or ov
    return out


def strata_of(metas, keep, coin_counts):
    ms = [m for m in metas if (m["wallet"], m["twap_id"]) in keep]
    ov = overlap_flags(ms)
    s_overlap = {(m["wallet"], m["twap_id"]): ("overlapping" if ov.get((m["wallet"], m["twap_id"])) else "isolated")
                 for m in ms}
    ns = sorted(m["notional_usd"] for m in ms)
    q = [ns[int(len(ns) * f)] for f in (0.25, 0.5, 0.75)] if ns else [0, 0, 0]

    def qname(x):
        return "Q1" if x < q[0] else "Q2" if x < q[1] else "Q3" if x < q[2] else "Q4"
    top = [c for c, _ in sorted(coin_counts.items(), key=lambda kv: -kv[1])[:5]]
    return {"overlap": s_overlap, "size_quartile": {(m["wallet"], m["twap_id"]): qname(m["notional_usd"]) for m in ms},
            "coin": {(m["wallet"], m["twap_id"]): (m["coin"] if m["coin"] in top else "other") for m in ms},
            "_meta": {"size_quartile_bounds_usd": q, "top_coins": top}}


def breakdown(ds, flagged):
    focus, keep = focus_for(ds, "drop_truncated", "with", flagged)
    coin_counts = Counter(f.coin for f in ds.fills if ds.meta_key_of_fid.get(f.fid) in keep)
    strata = strata_of(ds.metas, keep, coin_counts)
    meta_info = strata.pop("_meta")
    fills = sort_fills(ds.fills)
    rows = []
    for name, fn in HEURISTICS.items():
        for tau in TAUS:
            kw = {"window_s": tau} if name == "h2_fixed_window" else {"tau_s": tau}
            pred = fn(fills, **kw)
            for sname, smap in strata.items():
                for g in sorted(set(smap.values())):
                    ev = evaluate(ds.truth, pred, fills=fills,
                                  focus=lambda k, g=g, smap=smap: isinstance(k, tuple) and k[0] == "truth"
                                  and smap.get((k[1], k[2])) == g)
                    rows.append({"heuristic": name, "tau_s": tau, "stratum": sname, "group": g,
                                 **{k: ev[k] for k in ("n_truth_metaorders", "exact_recovered_rate", "merge_rate", "merge_twap_rate",
                                                       "contam_nontwap_rate", "split_rate", "notional_ratio_median",
                                                       "notional_ratio_p90", "notional_ratio_geomean", "notional_ratio_share_gt2",
                                                       "count_ratio_mean")}})
    for sname in ("overlap", "size_quartile", "coin"):
        print(f"  h1_gap_runs by {sname}:")
        for r in rows:
            if r["heuristic"] == "h1_gap_runs" and r["stratum"] == sname and r["tau_s"] in (60, 300, 1800, 3600):
                print(f"  {r['group']:<13}tau={r['tau_s']:>5.0f} n={r['n_truth_metaorders']:>5} "
                      f"exact {_pct(r['exact_recovered_rate']):>6} mrgTWAP {_pct(r['merge_twap_rate']):>6} "
                      f"contam {_pct(r['contam_nontwap_rate']):>6} split {_pct(r['split_rate']):>6} "
                      f"ntl_med {_f(r['notional_ratio_median'], 2)} p90 {_f(r['notional_ratio_p90'], 2)}")
    return {"strata_meta": meta_info, "n_scored_metaorders": len(keep), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="reconstruction-error benchmark against exchange labels")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="data directory (examples/sample for the bundled sample)")
    ap.add_argument("--out", type=Path, default=ROOT / "out", help="output directory (default out/)")
    ap.add_argument("--mode", choices=("both", "twap_only", "mixed"), default="both")
    ap.add_argument("--layers", default="random")
    ap.add_argument("--min-fills", type=int, default=1000)
    ap.add_argument("--breakdown", action="store_true",
                    help="stratify by overlap, size and coin (truncated dropped, pairs kept; one file per stream)")
    ap.add_argument("--no-bl", action="store_true", help="skip the Barone-Lillo rule evaluation")
    ap.add_argument("--log-trials", action="store_true", help="append these estimates to <data>/trials.jsonl")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layers = tuple(args.layers.split(","))
    modes = ("twap_only", "mixed") if args.mode == "both" else (args.mode,)
    bench_run = "BENCH-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    prefix = "reconstruction_" + ("" if layers == ("random",) else "_".join(layers) + "_")
    flagged = flagged_set(args.data)
    written = []
    for mode in modes:
        try:
            ds = load_dataset(args.data, mixed=(mode == "mixed"), layers=layers)
        except FileNotFoundError as e:
            sys.exit(str(e))
        n_noise = sum(1 for k in ds.truth.values() if is_noise_key(k))
        print(f"\n{mode}: run {bench_run}, data run {ds.run_id}, layers {layers}")
        print(f"accounts {len(ds.wallets)}, metaorders in window {len(ds.metas):,}, tape {len(ds.fills):,} fills "
              f"(non-TWAP {n_noise:,}), paired flags {len(flagged)}, dropped malformed rows {dict(ds.dropped) or 0}")
        if args.breakdown:
            res = {"benchmark_run": bench_run, "data_run_id": ds.run_id, "stream": mode, "scope": "drop_truncated/with",
                   "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **breakdown(ds, flagged)}
            dst = args.out / f"{prefix}breakdown_{mode}.json"
        else:
            cells = run_cells(ds, flagged, args.min_fills)
            res = {"benchmark_run": bench_run, "data_run_id": ds.run_id, "stream": mode, "layers": layers,
                   "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "sample": {"wallets": len(ds.wallets), "metaorders_in_window": len(ds.metas), "fills": len(ds.fills),
                              "noise_fills": n_noise, "dropped": dict(ds.dropped)},
                   "taus": list(TAUS), "n_flagged_metaorders": len(flagged),
                   "conventions": "cluster-then-select; same wallets/metaorders/windows in both streams; fills sorted by (time, fid)",
                   "cells": cells}
            if mode == "mixed" and not args.no_bl:
                res["bl_rule"] = bl_rule(ds, flagged)
            dst = args.out / f"{prefix}{mode}.json"
        dst.write_text(json.dumps(res, indent=1))
        written.append(dst)
        if args.log_trials and not args.breakdown:
            with (args.data / "trials.jsonl").open("a", encoding="utf-8") as f:
                for c in cells:
                    if c["state"] != "ok":
                        continue
                    h1 = [r for r in c["rows"] if r["heuristic"] == "h1_gap_runs"]
                    f.write(json.dumps({"ts": res["generated"], "script": "benchmark_reconstruction", "benchmark_run": bench_run,
                                        "stream": mode, "run_id": ds.run_id, "boundary": c["boundary"], "pairing": c["pairing"],
                                        "h1_f1_focus": {int(r["tau_s"]): r["pair_f1_focus"] for r in h1},
                                        "h1_notional_ratio_median": {int(r["tau_s"]): r["notional_ratio_median"] for r in h1}})
                            + "\n")
    print("\nwrote " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
