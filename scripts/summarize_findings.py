#!/usr/bin/env python3
"""
Pull the headline numbers out of the existing reports into data/findings_summary.json, together with
the root_hash of the data snapshot.

    python3 scripts/summarize_findings.py                             # reconstruction reports from out/, the rest from data/
    python3 scripts/summarize_findings.py --reconstruction-dir data   # reconstruction reports in data/
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
H1_KEYS = ("pair_f1", "pair_f1_focus", "jaccard_focus", "exact_recovered_rate", "merge_rate", "merge_twap_rate",
           "contam_nontwap_rate", "split_rate", "count_ratio_median", "count_ratio_p90", "count_ratio_mean",
           "notional_ratio_median", "notional_ratio_p90", "notional_ratio_geomean", "notional_ratio_share_gt2",
           "notional_ratio_share_gt10", "notional_ratio_aggregate")
TAUS = (60, 300, 900, 1800, 3600)


def load(name, base=OUT):
    p = base / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(r, boundary, pairing):
    c = next((x for x in r.get("cells", []) if x["boundary"] == boundary and x["pairing"] == pairing), None)
    return c if c and c.get("state") == "ok" else None


def h1_rows(rows):
    return {int(x["tau_s"]): x for x in rows if x["heuristic"] == "h1_gap_runs"}


def main():
    ap = argparse.ArgumentParser(description="collect the headline numbers from the reports")
    ap.add_argument("--reconstruction-dir", type=Path, default=ROOT / "out",
                    help="directory with the reconstruction reports (reconstruction_*.json)")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory for findings_summary.json (default data/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rep = load("collect_report.json")
    run_id = rep["run_id"]
    # snapshot manifest from the collection machine, None if it isn't there
    mf = load(f"manifest_{run_id}.json")
    summary = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": run_id,
                     "manifest_root_hash": mf and mf["root_hash"], "manifest_n_files": mf and mf["n_files"],
                     "reconstruction_dir": str(args.reconstruction_dir)}

    sr, st = rep["summary_random"], rep["summary_targeted"]
    summary["dataset"] = {"wallets_random": rep["wallets"]["random"], "wallets_targeted": rep["wallets"]["targeted"],
                          "metaorders_random": sr["all"]["n"], "metaorders_targeted": st["all"]["n"],
                          "random_boundary_excluded": sr["boundary_excluded"]["n"],
                          "random_fills_median": sr["all"]["fills_median"], "random_notional_median_usd": sr["all"]["notional_median"],
                          "random_notional_p90_usd": sr["all"]["notional_p90"], "random_n_ge_1m": sr["all"]["n_ge_1m"],
                          "coins_random": sr["all"]["coins"], "pagination_check": rep["pagination_check"]["state"],
                          "contamination_hit_rate": rep["contamination"]["hit_rate"]}

    recon = {}
    for stream in ("twap_only", "mixed"):
        r = load(f"reconstruction_{stream}.json", args.reconstruction_dir)
        if not r or "cells" not in r:
            continue
        main_cell = cell(r, "drop_truncated", "with")
        if not main_cell:
            continue
        h1 = h1_rows(main_cell["rows"])
        rows = {(x["heuristic"], int(x["tau_s"])): x for x in main_cell["rows"]}
        econ_keys = ("pair_f1_focus", "merge_twap_rate", "exact_recovered_rate")
        recon[stream] = {"benchmark_run": r["benchmark_run"], "sample": r["sample"],
                         "n_scored_metaorders": main_cell["n_scored_metaorders"], "n_scored_fills": main_cell["n_scored_fills"],
                         "h1": {tau: {k: h1[tau][k] for k in H1_KEYS} for tau in TAUS if tau in h1},
                         "h1_economic_truth": {int(x["tau_s"]): {k: x[k] for k in econ_keys}
                                               for x in main_cell.get("rows_economic", []) if x["heuristic"] == "h1_gap_runs"},
                         "h2_best_f1_focus": max(x["pair_f1_focus"] for (h, _), x in rows.items() if h == "h2_fixed_window"),
                         "h3_f1_focus_at_300": rows[("h3_direction_runs", 300)]["pair_f1_focus"]}
        if main_cell.get("rows_oid"):
            recon[stream]["h1_oid_truth_f1_focus"] = {int(x["tau_s"]): x["pair_f1_focus"] for x in main_cell["rows_oid"]
                                                      if x["heuristic"] == "h1_gap_runs"}
        without = cell(r, "drop_truncated", "without")
        if without:
            recon[stream]["without_pairs_h1_f1_focus"] = {tau: x["pair_f1_focus"] for tau, x in h1_rows(without["rows"]).items()}
        if r.get("bl_rule"):
            recon[stream]["bl_rule"] = r["bl_rule"]
        bd = load(f"reconstruction_breakdown_{stream}.json", args.reconstruction_dir)
        decomp = None
        if bd and bd.get("rows"):
            # Split the merge rate: "overlapping" = parallel same-side TWAPs (structural), "isolated" = the rest
            ov = {(x["group"], int(x["tau_s"])): x for x in bd["rows"]
                  if x["heuristic"] == "h1_gap_runs" and x["stratum"] == "overlap"}
            decomp = {}
            for tau in TAUS:
                a, b = ov.get(("overlapping", tau)), ov.get(("isolated", tau))
                if not a or not b:
                    decomp[tau] = None      # E.g. no parallel metaorders in the sample
                    continue
                n_a, n_b = a["n_truth_metaorders"], b["n_truth_metaorders"]
                total = (a["merge_twap_rate"] * n_a + b["merge_twap_rate"] * n_b) / (n_a + n_b)
                decomp[tau] = {"total_merge_twap_rate": round(total, 4),
                               "structural_share_of_metaorders": round(n_a / (n_a + n_b), 4),
                               "structural_merge_twap_rate": a["merge_twap_rate"], "isolated_merge_twap_rate": b["merge_twap_rate"],
                               "structural_contribution": round(a["merge_twap_rate"] * n_a / (n_a + n_b), 4),
                               "isolated_contribution": round(b["merge_twap_rate"] * n_b / (n_a + n_b), 4)}
            pick = ("n_truth_metaorders", "merge_twap_rate", "contam_nontwap_rate", "notional_ratio_median", "notional_ratio_p90")
            h1_300 = [x for x in bd["rows"] if x["heuristic"] == "h1_gap_runs" and x["tau_s"] == 300]
            recon[stream]["by_size_quartile_h1_300"] = {x["group"]: {k: x[k] for k in pick} for x in h1_300
                                                        if x["stratum"] == "size_quartile"}
            recon[stream]["by_coin_h1_300"] = {x["group"]: {k: x[k] for k in pick} for x in h1_300 if x["stratum"] == "coin"}
            recon[stream]["size_quartile_bounds_usd"] = bd["strata_meta"]["size_quartile_bounds_usd"]
        recon[stream]["merge_decomposition_h1"] = decomp
    summary["reconstruction"] = recon

    cr, cal = load("crossing_report.json"), load("pair_test_calibration.json")
    if cr:
        t = cr["strata"].get("targeted", {})
        summary["paired_metaorders"] = {
            "verdict": cr["verdict"], "calibration": cal and cal["verdict"], "calibration_problems": cal and cal["problems"],
            "targeted_pairs": len(t.get("pairs", [])), "cross_pairs": cr["cross"]["n_pairs"],
            "share_within_layer": cr["shares_within_layer"], "share_incl_cross": cr["shares_incl_cross"],
            "p_values_reference_only": {
                "targeted_block_flip": (t.get("block_flip") or {}).get("null", {}).get("p_value"),
                "targeted_multi_coin_max": ((t.get("multi_coin") or {}).get("max_coins") or {}).get("p_value"),
                "cross_block_flip": (cr["cross"].get("block_flip") or {}).get("null", {}).get("p_value")},
            "shared_tids": {"tids_total": cr["shared_tids"].get("tids_total"),
                            "tids_shared": cr["shared_tids"].get("tids_shared"),
                            "role": cr["shared_tids"].get("role")}}

    imp, pw = load("impact_report.json"), load("impact_power.json")
    if imp:
        p = imp["results"]["truth"]["random/with/min3bars"]["I_tmp"]["pooled"]
        large = {k: v["I_tmp"]["pooled"] for k, v in imp["results"]["truth"].items() if "large" in k}
        summary["impact_law"] = {
            "state": p["state"], "n": p["n"], "n_wallets": p.get("n_wallets"), "x_median": p.get("x_median"),
            "I_mean_bp": p.get("I_mean_bp"), "I_sd_bp": p.get("I_sd_bp"), "I_pos_frac": p.get("I_pos_frac"),
            "beta_nls": (p.get("nls") or {}).get("beta"), "ci95": (p.get("nls_bootstrap") or {}).get("ci95"),
            "large_subsamples": {k: {"state": q["state"], "n": q.get("n"), "beta": (q.get("nls") or {}).get("beta"),
                                     "ci95": (q.get("nls_bootstrap") or {}).get("ci95"), "why": q.get("why")}
                                 for k, q in large.items()},
            "power": pw and {"observed": pw["observed"], "thresholds": pw["thresholds"],
                             "metaorders_per_snapshot_350": pw["metaorders_per_snapshot_350"]}}

    dst = args.out / "findings_summary.json"
    dst.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "reconstruction"}, indent=1))
    for stream, w in recon.items():
        h = w["h1"]
        for tau in (300, 1800):
            if tau in h:
                x = h[tau]
                print(f"\n[{stream}] metaorders {w['n_scored_metaorders']}, h1@{tau} F1(focus) {x['pair_f1_focus']} "
                      f"Jaccard {x['jaccard_focus']} exact {x['exact_recovered_rate']:.1%} merged-TWAP {x['merge_twap_rate']:.1%} "
                      f"non-TWAP-contam {x['contam_nontwap_rate']:.1%} notional ratio median {x['notional_ratio_median']} "
                      f"p90 {x['notional_ratio_p90']} share >2x {x['notional_ratio_share_gt2']:.1%}")
        for tau, d in (w.get("merge_decomposition_h1") or {}).items():
            if d is None:
                print(f"   tau={tau}: breakdown missing, cannot decompose")
                continue
            print(f"   tau={tau}: total merged-TWAP {d['total_merge_twap_rate']:.1%} = parallel {d['structural_contribution']:.1%}"
                  f" ({d['structural_share_of_metaorders']:.1%} of metaorders x merge rate {d['structural_merge_twap_rate']:.0%})"
                  f" + isolated {d['isolated_contribution']:.1%} (merge rate {d['isolated_merge_twap_rate']:.1%})")
        if w.get("bl_rule"):
            b = w["bl_rule"]
            print(f"   [BL rule] covers {b['coverage_by_count']:.1%} of metaorders ({b['coverage_by_notional']:.1%} of notional), "
                  f"exact recovery on covered {b['metrics_on_covered']['exact_recovered_rate']:.1%}")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
