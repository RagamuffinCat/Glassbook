#!/usr/bin/env python3
"""
Flag paired metaorders: opposite-side metaorders of similar size, started close together on the
same coin by different wallets. Pairs like that net out in price and would pull the impact
exponent down, so the impact law is reported with and without them.

Are opposite sides more common than chance among the size/time coincidences? Shuffling sides
within a coin doesn't answer that: wallets often split one order into several same-side TWAPs a
few seconds apart, and the shuffle turns those into fake opposite pairs. The block-flip null
groups each wallet's metaorders on a coin into blocks (starts chained within --start-tol) and
flips whole blocks instead. The side shuffle is reported too. The multi-coin test asks whether
the same two wallets match on 2 or more coins, against a within-coin wallet shuffle.

Random and targeted layers are tested separately. If data/pair_test_calibration.json says
"fail", the verdict becomes calibration_failed.

    python3 scripts/detect_crossing.py
    python3 scripts/detect_crossing.py --size-tol 0.02 --start-tol 120
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(OUT.parent))

from glassbook.pairs import null_burst_flip, offsetting_pairs, perm_dist, read_jsonl, record_trial


def shared_tids(slices):
    """Trade ids seen in the TWAP fills of two or more wallets.

    Recorded only. TWAP children are always takers, so two wallets' TWAP fills can't share a trade.
    A shared tid would mean a data problem such as tid reuse.
    """
    by_tid = defaultdict(set)
    tid_meta = {}
    for s in slices:
        tid = s.get("tid")
        if tid is None:
            continue
        by_tid[int(tid)].add(s["wallet"])
        tid_meta[int(tid)] = (s.get("coin"), s.get("twap_id"), s["wallet"])

    pairs = defaultdict(int)
    coins = defaultdict(set)
    for tid, ws in by_tid.items():
        if len(ws) < 2:
            continue
        for a, b in combinations(sorted(ws), 2):
            pairs[(a, b)] += 1
            coins[(a, b)].add(tid_meta[tid][0])

    return {
        "tids_total": len(by_tid),
        "tids_shared": sum(1 for ws in by_tid.values() if len(ws) > 1),
        "wallet_pairs": sorted(
            ({"a": a, "b": b, "shared_tids": n, "coins": sorted(coins[(a, b)])}
             for (a, b), n in pairs.items()),
            key=lambda d: -d["shared_tids"]),
    }


def _shuffle_sides_within_coin(metas, rng):
    by_coin = defaultdict(list)
    for m in metas:
        by_coin[m.get("coin")].append(m)
    out = []
    for coin, rows in by_coin.items():
        sides = [r.get("side") for r in rows]
        rng.shuffle(sides)
        out += [{**r, "side": sd} for r, sd in zip(rows, sides)]
    return out


def null_baseline(metas, size_tol, start_tol_s, reps=200, seed=20260920):
    """Side-shuffle null, reported next to the block flip for comparison."""
    rng = random.Random(seed)
    counts = [len(offsetting_pairs(_shuffle_sides_within_coin(metas, rng),
                                   size_tol, start_tol_s)["pairs"]) for _ in range(reps)]
    counts.sort()
    n = len(counts)
    return {"reps": reps, "mean": round(sum(counts) / n, 2), "p50": counts[n // 2],
            "p95": counts[min(n - 1, int(.95 * n))], "max": counts[-1]}


def p_value(observed, null, metas, size_tol, start_tol_s, reps=200, seed=20260920):
    """Side-shuffle right-tail permutation p-value with add-one smoothing."""
    # own draw (seed + 1), not null_baseline's counts. reusing them is faster but changes the
    # published p-values
    rng = random.Random(seed + 1)
    ge = 0
    for _ in range(reps):
        if len(offsetting_pairs(_shuffle_sides_within_coin(metas, rng),
                                size_tol, start_tol_s)["pairs"]) >= observed:
            ge += 1
    return round((ge + 1) / (reps + 1), 4)


def same_side_split_pairs(metas, size_tol, start_tol_s):
    """Same-wallet same-side pairs within tolerance. These are the split orders that inflate the side-shuffle null."""
    ps = offsetting_pairs(metas, size_tol, start_tol_s, require_opposite=False)["pairs"]
    return sum(1 for p in ps if p["same_wallet"] and p["same_side"])


def _rep_counts(ps):
    by = defaultdict(set)
    for x in ps:
        if x["same_wallet"]:
            continue
        k = tuple(sorted((x["a"]["wallet"], x["b"]["wallet"])))
        by[k].add(x["coin"])
    return by


def pair_repetition(pairs, metas, size_tol, start_tol_s, reps=200, seed=20260920):
    """Wallet pairs matched on 2+ coins, plus max_coins for any one pair, vs. a within-coin wallet shuffle.
    The only pair statistic with any power when pairs are sparse."""
    obs = _rep_counts(pairs)
    obs_multi = {k: sorted(v) for k, v in obs.items() if len(v) >= 2}
    obs_max = max((len(v) for v in obs.values()), default=0)

    rng = random.Random(seed)
    null_multi, null_max = [], []
    for _ in range(reps):
        # rebuilt each rep, rows get replaced below
        by_coin = defaultdict(list)
        for m in metas:
            by_coin[m.get("coin")].append(m)
        shuffled = []
        for coin, rows in by_coin.items():
            ws = [r["wallet"] for r in rows]
            rng.shuffle(ws)
            shuffled += [{**r, "wallet": w} for r, w in zip(rows, ws)]
        nb = _rep_counts(offsetting_pairs(shuffled, size_tol, start_tol_s)["pairs"])
        null_multi.append(sum(1 for v in nb.values() if len(v) >= 2))
        null_max.append(max((len(v) for v in nb.values()), default=0))

    null_multi.sort()
    n = len(null_multi)
    ge = sum(1 for x in null_multi if x >= len(obs_multi))
    # zero-variance null (e.g. only two wallets in the stratum) -> nothing to test
    degenerate = n > 0 and null_multi[0] == null_multi[-1]
    n_wallets = len({m["wallet"] for m in metas})
    n_coins = len({m.get("coin") for m in metas})
    reason = ("only 1 coin in this stratum; cross-coin repetition is undefined" if n_coins < 2
              else f"only {n_wallets} wallets in this stratum; the permutation has no degrees of freedom" if n_wallets < 4
              else "null distribution has zero variance")
    dmax = perm_dist(null_max, obs_max)
    return {
        "degenerate": degenerate,
        "degenerate_reason": reason if degenerate else None,
        "n_wallets": n_wallets,
        "n_coins": n_coins,
        "wallet_pairs_total": len(obs),
        "wallet_pairs_multicoin": len(obs_multi),
        "detail": [{"a": k[0], "b": k[1], "coins": v} for k, v in obs_multi.items()],
        "null_mean": round(sum(null_multi) / n, 3) if n else 0,
        "null_p95": null_multi[min(n - 1, int(.95 * n))] if n else 0,
        "p_value": round((ge + 1) / (n + 1), 4) if n else 1.0,
        "max_coins": {"observed": obs_max, "null": dmax,
                      "degenerate": bool(dmax.get("degenerate")) if n else True,
                      "p_value": dmax.get("p_value", 1.0)},
    }


def rule_of_three(n):
    # 3/n = 95% upper bound on the rate after 0 hits in n trials
    return 3.0 / n if n else 1.0


def power_note(metas, interest_usd=1_000_000):
    """Whether the sample reaches the notional tail at all. With few metaorders above interest_usd,
    0 hits says little."""
    ns = sorted(m["notional_usd"] for m in metas if m.get("notional_usd"))
    big = [x for x in ns if x >= interest_usd]
    return {"n": len(ns), "max_usd": round(ns[-1], 2) if ns else 0,
            "p90_usd": round(ns[min(len(ns) - 1, int(.9 * len(ns)))], 2) if ns else 0,
            "n_above_interest": len(big), "interest_usd": interest_usd,
            "underpowered_for_tail": len(big) < 5}


def pair_observability(rep):
    """Pair visibility at random sampling fraction f: ~f^2 for cross-wallet pairs, ~f for same-wallet ones."""
    pool = (rep.get("pool_profile") or {}).get("pool_size") or 0
    n = (rep.get("wallets") or {}).get("random") or 0
    f = n / pool if pool else 0.0
    return {"sampling_frac": round(f, 3), "cross_wallet_visible": round(f * f, 4),
            "self_pair_visible": round(f, 3)}


def layer_shares(metas, flagged):
    """Per-stratum notional share of flagged metaorders. flagged may include cross-stratum pairs."""
    out = {}
    for tag in sorted({m.get("sampling") for m in metas}):
        rows = [m for m in metas if m.get("sampling") == tag]
        tot = sum(m["notional_usd"] for m in rows)
        fl = sum(m["notional_usd"] for m in rows if (m["wallet"], m["twap_id"]) in flagged)
        out[tag] = {"n": len(rows), "flagged": sum(1 for m in rows if (m["wallet"], m["twap_id"]) in flagged),
                    "share_by_notional": round(fl / tot, 4) if tot else 0,
                    "flagged_notional_usd": round(fl, 2), "total_notional_usd": round(tot, 2)}
    return out


def grade(share):
    """Grade the targeted stratum's flagged notional share, cross flags included."""
    if share is None:
        return "ungradable"
    if share >= 0.20:
        return "beta_with_and_without"
    if share >= 0.05:
        return "sensitivity_check"
    return "share_only"


def main():
    ap = argparse.ArgumentParser(description="detect paired metaorders")
    ap.add_argument("--size-tol", type=float, default=0.05,
                    help="tolerance on |log size ratio| (default 0.05, about 5%%)")
    ap.add_argument("--start-tol", type=float, default=60,
                    help="max start-time difference in seconds (default 60)")
    ap.add_argument("--top", type=int, default=12, help="number of pairs to print")
    ap.add_argument("--null-reps", type=int, default=200, help="permutation test repetitions")
    ap.add_argument("--no-null", action="store_true", help="skip the permutation tests (fast, but no p-values)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where crossing_report.json goes (trials.jsonl too), default data/")
    ap.add_argument("--seed", type=int, default=0, help="offset added to the default permutation seeds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trials = args.out / "trials.jsonl"

    try:
        metas = read_jsonl(OUT / "metaorders.jsonl")
    except FileNotFoundError as e:
        sys.exit(str(e))
    rep = json.loads((OUT / "collect_report.json").read_text())
    obs = pair_observability(rep)
    n_wallets_all = len({m["wallet"] for m in metas})
    print(f"run {rep['run_id']}")
    print(f"random-stratum sampling fraction {obs['sampling_frac']:.1%}; share of pairs visible: cross-wallet ~"
          f"{obs['cross_wallet_visible']:.1%}, same-wallet ~{obs['self_pair_visible']:.1%}")
    slices_p = OUT / "slices.jsonl"
    has_slices = slices_p.exists() or slices_p.with_name(slices_p.name + ".gz").exists()
    slices = read_jsonl(slices_p) if has_slices else []

    print("\nTrade ids shared by 2+ wallets (none expected)")
    if not slices:
        print("  no slices.jsonl(.gz); skipped")
        a = {"state": "skipped_no_slices", "role": "record_only"}
    else:
        a = shared_tids(slices)
        a["role"] = "record_only"
        print(f"  trade ids in TWAP fills {a['tids_total']:,}, shared by 2+ wallets {a['tids_shared']:,}"
              f" ({a['tids_shared']/max(1,a['tids_total']):.2%})")
        if a["wallet_pairs"]:
            print(f"\n  {'wallet A':<14}{'wallet B':<14}{'shared':>10}  coins")
            for p in a["wallet_pairs"][:args.top]:
                print(f"  {p['a'][:12]:<14}{p['b'][:12]:<14}{p['shared_tids']:>10}  "
                      f"{','.join(p['coins'][:5])}")
            print("\n  warning: TWAP fills on both sides of a trade shouldn't happen, tid reuse?")
        else:
            print(f"  no shared trade id across {n_wallets_all} wallets, as expected")

    print(f"\nOpposite-side pairs vs. chance (size within {args.size_tol:.0%}, starts within {args.start_tol:.0f}s)\n"
          "  side-shuffle null: sides shuffled within each coin; block-flip null: each wallet's block of nearby orders flipped as a unit")
    strata = {}
    for tag in ("random", "targeted"):
        rows = [m for m in metas if m.get("sampling") == tag]
        if not rows:
            continue
        st = {**offsetting_pairs(rows, args.size_tol, args.start_tol), "power": power_note(rows)}
        st["split_same_side_pairs"] = same_side_split_pairs(rows, args.size_tol, args.start_tol)
        st["coincidences_any_side"] = sum(
            1 for p in offsetting_pairs(rows, args.size_tol, args.start_tol,
                                        require_opposite=False)["pairs"] if not p["same_wallet"])
        if not args.no_null:
            st["null"] = null_baseline(rows, args.size_tol, args.start_tol, args.null_reps,
                                       seed=20260920 + args.seed)
            st["p_value"] = p_value(len(st["pairs"]), st["null"], rows,
                                    args.size_tol, args.start_tol, args.null_reps,
                                    seed=20260920 + args.seed)
            st["excess_pairs"] = round(len(st["pairs"]) - st["null"]["mean"], 2)
            st["null_degenerate"] = st["null"]["p95"] == st["null"]["p50"] == int(st["null"]["mean"])
            # observed < null mean / 3 -> side-shuffle null misspecified (split orders, module docstring)
            st["null_misspecified"] = (st["null"]["mean"] > 0 and
                                       len(st["pairs"]) < st["null"]["mean"] / 3)
            # Cross-wallet pairs only, same-wallet ones never change under a block flip
            counts = null_burst_flip(rows, args.size_tol, args.start_tol,
                                     lambda p: not p["same_wallet"], args.null_reps,
                                     seed=20260923 + args.seed)
            st["block_flip"] = {"observed_cross_wallet": st["cross_wallet_pairs"],
                                "null": perm_dist(counts, st["cross_wallet_pairs"])}
            st["multi_coin"] = pair_repetition(st["pairs"], rows, args.size_tol, args.start_tol,
                                               args.null_reps, seed=20260920 + args.seed)
        strata[tag] = st

    # Cross: one side random, the other targeted
    b_all = offsetting_pairs(metas, args.size_tol, args.start_tol)
    cross_pairs = [p for p in b_all["pairs"] if p["cross_layer"]]
    cross = {"pairs": cross_pairs, "n_pairs": len(cross_pairs),
                   "notional_both_sides": round(sum(p["a"]["notional"] + p["b"]["notional"]
                                                    for p in cross_pairs), 2),
                   "wallet_pairs": sorted({tuple(sorted((p["a"]["wallet"], p["b"]["wallet"])))
                                           for p in cross_pairs})}
    if cross_pairs and not args.no_null:
        counts = null_burst_flip(metas, args.size_tol, args.start_tol,
                                 lambda p: p["cross_layer"], args.null_reps,
                                 seed=20260923 + args.seed)
        cross["block_flip"] = {"observed": len(cross_pairs), "null": perm_dist(counts, len(cross_pairs))}

    # shares_all also counts metaorders flagged only by cross-stratum pairs
    flagged_within = {(p[k]["wallet"], p[k]["twap_id"])
                      for st in strata.values() for p in st["pairs"] for k in ("a", "b")}
    flagged_all = flagged_within | {(p[k]["wallet"], p[k]["twap_id"]) for p in cross_pairs for k in ("a", "b")}
    shares_within = layer_shares(metas, flagged_within)
    shares_all = layer_shares(metas, flagged_all)

    def f1(x):
        return "n/a" if x is None else f"{x:.1f}"

    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"

    # within / +cross: flagged notional share without and with the cross-stratum flags
    print(f"\n  {'stratum':<10}{'meta':>6}{'pairs':>5}{'x-wal':>7}{'shuf null':>10}{'shuf p':>7}"
          f"{'flip null':>10}{'flip p':>7}{'within':>8}{'+cross':>8}{'max notional':>13}")
    for tag, st in strata.items():
        nl, bf = st.get("null"), st.get("block_flip")
        shm = nl["mean"] if nl else None
        shp = st.get("p_value") if nl else None
        bfm = bf["null"]["mean"] if bf else None
        bfp = bf["null"]["p_value"] if bf else None
        sw = shares_within.get(tag, {}).get("share_by_notional", 0)
        sa = shares_all.get(tag, {}).get("share_by_notional", 0)
        print(f"  {tag:<10}{st['power']['n']:>6}{len(st['pairs']):>5}{st['cross_wallet_pairs']:>7}"
              f"{f1(shm):>10}{f3(shp):>7}{f1(bfm):>10}{f3(bfp):>7}{sw:>8.2%}{sa:>8.2%}"
              f"{st['power']['max_usd']:>13,.0f}")
    cv = cross.get("block_flip")
    print(f"  {'cross':<10}{'n/a':>6}{cross['n_pairs']:>5}{cross['n_pairs']:>7}{'n/a':>10}{'n/a':>7}"
          f"{f1(cv['null']['mean'] if cv else None):>10}{f3(cv['null']['p_value'] if cv else None):>7}"
          f"{'':>8}{'':>8}{'':>13}  wallet pairs {len(cross['wallet_pairs'])}")

    def pv(d):
        return "untestable" if d.get("degenerate") else f"p={d['p_value']:.3f}"

    for tag, st in strata.items():
        pw, nl, bf = st["power"], st.get("null"), st.get("block_flip")
        if not st["pairs"]:
            print(f"\n  [{tag}] 0 pairs, 95% upper bound on the rate {rule_of_three(pw['n']):.2%}"
                  + (f" (but only ~{obs['cross_wallet_visible']:.1%} of cross-wallet pairs are visible)"
                     if tag == "random" else ""))
            if pw["underpowered_for_tail"]:
                print(f"  [{tag}] only {pw['n_above_interest']} metaorders of ${pw['interest_usd']:,.0f} or more")
            continue
        print(f"\n  [{tag}] {len(st['pairs'])} pairs (cross-wallet {st['cross_wallet_pairs']}, "
              f"same-wallet {st['same_wallet_pairs']})")
        if nl:
            note = (f", misspecified ({st['split_same_side_pairs']} same-side split pairs)" if st.get("null_misspecified")
                    else ", degenerate" if st.get("null_degenerate") else "")
            print(f"      side shuffle  null {nl['mean']:.1f}, p={st['p_value']:.3f}{note}")
        if bf:
            d = bf["null"]
            print(f"      block flip    {st['cross_wallet_pairs']} of {st['coincidences_any_side']} cross-wallet "
                  f"matches opposite, null {d['mean']:.2f} (p95 {d['p95']}), {pv(d)}")
        c = st.get("multi_coin")
        if c and c["wallet_pairs_total"]:
            mx = c["max_coins"]
            print(f"      multi-coin    {c['wallet_pairs_multicoin']} of {c['wallet_pairs_total']} wallet pairs on 2+ coins, "
                  f"null {c['null_mean']:.2f}, " + (f"untestable ({c['degenerate_reason']})" if c["degenerate"]
                                                     else f"p={c['p_value']:.3f}"))
            print(f"      max coins     {mx['observed']}, null {mx['null'].get('mean', 0):.2f} "
                  f"(p95 {mx['null'].get('p95', 0)}), {pv(mx)}")
            for d in c["detail"][:5]:
                print(f"        {d['a'][:12]} / {d['b'][:12]}  coins {','.join(d['coins'])}")

    if cross_pairs:
        print(f"\n  [cross] random/targeted pairs {cross['n_pairs']}, notional of both sides "
              f"${cross['notional_both_sides']:,.0f}, wallet pairs {len(cross['wallet_pairs'])}")
        for wp in cross["wallet_pairs"][:5]:
            print(f"        {wp[0][:12]} / {wp[1][:12]}")
        if cross.get("block_flip"):
            d = cross["block_flip"]["null"]
            print(f"      block flip    null {d['mean']:.2f} (p95 {d['p95']}), {pv(d)}")

    print("\n  flagged notional share by stratum (within only / incl. cross): " + ", ".join(
        f"{t} {shares_within[t]['share_by_notional']:.2%} / {shares_all[t]['share_by_notional']:.2%}"
        for t in shares_all))
    if b_all["pairs"]:
        print(f"\n  {'coin':<12}{'dt(s)':>7}{'ratio':>8}{'notional A':>13}{'notional B':>13}  wallets")
        for p in b_all["pairs"][:args.top]:
            tag = "same-wallet" if p["same_wallet"] else ("cross-stratum" if p["cross_layer"] else "")
            print(f"  {str(p['coin']):<12}{p['dt_s']:>7.0f}{p['size_ratio']:>8.3f}"
                  f"{p['a']['notional']:>13,.0f}{p['b']['notional']:>13,.0f}  "
                  f"{p['a']['wallet'][:10]}/{p['b']['wallet'][:10]} {tag}")

    print("")
    calib_p = OUT / "pair_test_calibration.json"
    calib = json.loads(calib_p.read_text()) if calib_p.exists() else None
    calib_failed = bool(calib and calib["verdict"] == "fail")
    if calib_failed:
        print("  calibration failed (" + "; ".join(calib.get("problems", [])) + ")")
    rs = strata.get("random") or next(iter(strata.values()))
    sig = []
    for t, st in strata.items():
        bf = st.get("block_flip")
        if bf and not bf["null"].get("degenerate") and bf["null"]["p_value"] <= 0.05:
            sig.append(f"{t}/block_flip")
        c = st.get("multi_coin") or {}
        if c and not c.get("degenerate", True) and c.get("p_value", 1.0) <= 0.05:
            sig.append(f"{t}/multi_coin")
        mx = c.get("max_coins") or {}
        if mx and not mx.get("degenerate", True) and mx.get("p_value", 1.0) <= 0.05:
            sig.append(f"{t}/multi_coin_max")
    cb = cross.get("block_flip")
    if cb and not cb["null"].get("degenerate") and cb["null"]["p_value"] <= 0.05:
        sig.append("cross/block_flip")

    t_share = shares_all.get("targeted", {}).get("share_by_notional") if "targeted" in strata else None
    r_share = shares_all.get("random", {}).get("share_by_notional") if "random" in strata else None
    g = grade(t_share)
    if "targeted" not in strata or rs["power"]["underpowered_for_tail"] and not sig:
        verdict = "underpowered"
        print("  verdict: underpowered, too little of the tail in the sample")
    else:
        verdict = "calibration_failed" if calib_failed else g
        print(f"  significant: {', '.join(sig) if sig else 'none'}")
        print(f"  flagged notional share incl. cross: random {r_share:.2%}, targeted {t_share:.2%} -> grade {g}")

    params = {k: v for k, v in vars(args).items() if k != "out"}
    result = {"params": params, "verdict": verdict, "significant": sig,
              "run_id": rep["run_id"], "observability": obs,
              "shared_tids": a, "all_pairs": b_all["pairs"], "cross": cross,
              "shares_within_layer": shares_within, "shares_incl_cross": shares_all,
              "strata": strata}
    dst = args.out / "crossing_report.json"
    dst.write_text(json.dumps(result, indent=2))
    # One trials row per test
    base = {"script": "detect_crossing", "run_id": rep["run_id"],
            "params": params, "verdict": verdict}
    n_rec = 0
    for t, st in strata.items():
        tests = {"side_shuffle": st.get("p_value"),
                 "block_flip": (st.get("block_flip") or {}).get("null", {}).get("p_value"),
                 "multi_coin": (st.get("multi_coin") or {}).get("p_value"),
                 "multi_coin_max": ((st.get("multi_coin") or {}).get("max_coins") or {}).get("p_value")}
        for name, pv in tests.items():
            if pv is None:
                continue
            record_trial(trials, {**base, "stratum": t, "test": name, "p_value": pv,
                                  "share_within": shares_within.get(t, {}).get("share_by_notional"),
                                  "share_incl_cross": shares_all.get(t, {}).get("share_by_notional")})
            n_rec += 1
    if cross.get("block_flip"):
        record_trial(trials, {**base, "stratum": "cross", "test": "block_flip",
                              "p_value": cross["block_flip"]["null"]["p_value"], "n_pairs": cross["n_pairs"]})
        n_rec += 1
    print(f"\nwrote {dst}; appended {n_rec} rows to {trials} (now {sum(1 for _ in trials.open()) if trials.exists() else 0} rows)\n")


if __name__ == "__main__":
    main()
