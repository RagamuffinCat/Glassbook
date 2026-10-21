"""Paired metaorders: opposite-side metaorders of similar size started close together on one coin.

Pair finder + block-flip null, shared by detect_crossing.py and calibrate_pair_tests.py (must agree exactly).
"""

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .metaorder.dataset import open_jsonl

FLIP = {"B": "A", "A": "B"}


def read_jsonl(path):
    """Rows of <name>.jsonl, or of <name>.jsonl.gz."""
    return [json.loads(line) for line in open_jsonl(Path(path)) if line.strip()]


def offsetting_pairs(metas: list[dict], size_tol: float, start_tol_s: float,
                     require_opposite: bool = True) -> dict:
    """Same-coin pairs with similar size and close starts. require_opposite=False also keeps same-side pairs."""
    by_coin: defaultdict[str, list[dict]] = defaultdict(list)
    for m in metas:
        if m.get("size") and m.get("coin"):
            by_coin[m["coin"]].append(m)

    found = []
    flagged: set[tuple] = set()
    for coin, rows in by_coin.items():
        rows.sort(key=lambda m: m["t_start_ms"])
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                dt = (b["t_start_ms"] - a["t_start_ms"]) / 1000
                if dt > start_tol_s:
                    break                      # rows sorted by start, everything after is further away
                if require_opposite and a.get("side") == b.get("side"):
                    continue
                sa, sb = abs(a["size"]), abs(b["size"])
                if sa <= 0 or sb <= 0:
                    continue
                if abs(math.log(sa / sb)) > size_tol:
                    continue
                found.append({
                    "coin": coin, "dt_s": round(dt, 1),
                    "size_ratio": round(sa / sb, 4),
                    "a": {"wallet": a["wallet"], "twap_id": a["twap_id"], "side": a["side"],
                          "notional": a["notional_usd"], "dur_s": a["duration_s"],
                          "sampling": a.get("sampling")},
                    "b": {"wallet": b["wallet"], "twap_id": b["twap_id"], "side": b["side"],
                          "notional": b["notional_usd"], "dur_s": b["duration_s"],
                          "sampling": b.get("sampling")},
                    "same_wallet": a["wallet"] == b["wallet"],
                    "same_side": a.get("side") == b.get("side"),
                    "cross_layer": a.get("sampling") != b.get("sampling"),
                })
                flagged.add((a["wallet"], a["twap_id"]))
                flagged.add((b["wallet"], b["twap_id"]))

    total_n = len(metas)
    total_notional = sum(m["notional_usd"] for m in metas)
    fl_notional = sum(m["notional_usd"] for m in metas
                      if (m["wallet"], m["twap_id"]) in flagged)
    return {
        "pairs": sorted(found, key=lambda d: -(d["a"]["notional"] + d["b"]["notional"])),
        "metaorders_flagged": len(flagged),
        "metaorders_total": total_n,
        "share_by_count": round(len(flagged) / total_n, 4) if total_n else 0,
        "share_by_notional": round(fl_notional / total_notional, 4) if total_notional else 0,
        "flagged_notional_usd": round(fl_notional, 2),
        "total_notional_usd": round(total_notional, 2),
        "same_wallet_pairs": sum(1 for f in found if f["same_wallet"]),
        "cross_wallet_pairs": sum(1 for f in found if not f["same_wallet"]),
    }


def perm_dist(counts, observed):
    """Summary of a permutation null and the right-tail p-value of observed, with add-one smoothing."""
    counts = sorted(counts)
    n = len(counts)
    if not n:
        return {"reps": 0}
    ge = sum(1 for x in counts if x >= observed)      # ties count as >= (conservative)
    return {"reps": n, "mean": round(sum(counts) / n, 2), "p50": counts[n // 2],
            "p95": counts[min(n - 1, int(.95 * n))], "max": counts[-1],
            "degenerate": counts[0] == counts[-1],
            "p_value": round((ge + 1) / (n + 1), 4)}


def split_bursts(metas, start_tol_s):
    """Blocks of one wallet's metaorders on a coin whose starts chain within start_tol.
    Any same-wallet pair within start_tol lands in a single block."""
    by = defaultdict(list)
    for m in metas:
        by[(m["wallet"], m.get("coin"))].append(m)
    out = []
    for rows in by.values():
        rows.sort(key=lambda m: m["t_start_ms"])
        cur = [rows[0]]
        for m in rows[1:]:
            if (m["t_start_ms"] - cur[-1]["t_start_ms"]) / 1000 <= start_tol_s:
                cur.append(m)
            else:
                out.append(cur)
                cur = [m]
        out.append(cur)
    return out


def null_burst_flip(metas, size_tol, start_tol_s, pair_filter, reps=200, seed=20260923):
    """Block-flip null: flip each block's side with probability 1/2. Returns per-rep counts of pairs passing pair_filter.
    Same-wallet pairs sit inside one block, so a flip never changes them."""
    rng = random.Random(seed)
    blocks = split_bursts(metas, start_tol_s)
    counts = []
    for _ in range(reps):
        shuffled = []
        for b in blocks:
            flip = rng.random() < 0.5
            for m in b:
                sd = m.get("side")
                shuffled.append({**m, "side": FLIP.get(sd, sd) if flip else sd})
        ps = offsetting_pairs(shuffled, size_tol, start_tol_s)["pairs"]
        counts.append(sum(1 for p in ps if pair_filter(p)))
    return counts


def record_trial(trials, entry):
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with Path(trials).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def calibration_problems(res, fpr_max):
    """Reasons the calibration of the block-flip test fails; empty means pass."""
    problems = []
    for k in ("false_positive_targeted", "false_positive_cross"):
        if res[k]["fpr_005"] > fpr_max:
            problems.append(f"{k} FPR(0.05)={res[k]['fpr_005']} > {fpr_max}: the block-flip test rejects too often")
    for k in ("reseed_targeted", "reseed_cross"):
        if res[k]["degenerate"]:
            problems.append(f"{k} null has zero variance: untestable")
        elif res[k]["p_value"] > 0.05:
            problems.append(f"{k} independent-seed p={res[k]['p_value']} > 0.05: block-flip result unstable")
    for k, r in res["coincidence"].items():
        if r["degenerate"]:
            problems.append(f"coincidence {k} null has zero variance: untestable")
        elif r["p_value"] > 0.05:
            problems.append(f"coincidence {k} p={r['p_value']} > 0.05: coincidence not significant")
    return problems
