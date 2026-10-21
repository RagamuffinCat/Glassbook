#!/usr/bin/env python3
"""
Collect the non-TWAP fills (`userFillsByTime`) of each sampled wallet. TWAP fills alone only show
merging and splitting between metaorders, which is a lower bound on the reconstruction error.
Writes data/mixed_fills.jsonl and data/mixed_report.json.

userFillsByTime doesn't return TWAP child fills at all (0 of the TWAP tids show up in it), so the
mixed stream is the union of this output and slices.jsonl. The overlap check makes sure the two
really don't share anything.

    python3 scripts/collect_mixed.py
    python3 scripts/collect_mixed.py --resume
    python3 scripts/collect_mixed.py --recheck  # offline: recompute the overlap check from the checkpoint
"""

import argparse
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"
CKPT = OUT / "checkpoints"

from glassbook.checkpoint import Checkpoint, atomic_jsonl
from glassbook.hl.client import APIRequestFailed, default_client
from glassbook.hl.paging import fill_tid, fill_time, paginate_adaptive

ZERO = "0x" + "0" * 64          # The whole hash, stream.py only needs the prefix
BREAKER = 3


def make_fills_fetch(client):
    def fetch(user, start_ms, end_ms):
        r = client.post({"type": "userFillsByTime", "user": user,
                         "startTime": int(start_ms), "endTime": int(end_ms)})
        if not isinstance(r, list):
            raise APIRequestFailed(f"non-list response: {str(r)[:100]}")
        return r
    return fetch


def check_source_overlap(fills, twap_rows, t_lo, t_hi):
    """Source overlap check ("source_overlap" in the report): userFills and TWAP fills must share no tid.

    A shared tid would make that fill's label ambiguous, and with no fills from one of the sources in
    [t_lo, t_hi] there is nothing to compare. Also counts zero-hash userFills by dir and crossed; in
    our data these were mostly resting orders hit by another wallet's TWAP, plus Spot Dust Conversion.
    """
    fill_tids = {int(f["tid"]) for f in fills if f.get("tid") is not None}
    twap_all = {int(r["tid"]) for r in twap_rows if r.get("tid") is not None}
    overlap = fill_tids & twap_all
    n_twap_win = sum(1 for r in twap_rows if r.get("tid") is not None and t_lo <= int(r["time"]) <= t_hi)
    n_fill_win = sum(1 for f in fills if f.get("tid") is not None and t_lo <= int(f.get("time", 0)) <= t_hi)
    zero = [f for f in fills if str(f.get("hash", "")).startswith(ZERO) and f.get("tid") is not None]
    if overlap:
        state = "fail"
    elif n_twap_win == 0 or n_fill_win == 0:
        state = "inconclusive"
    else:
        state = "pass"
    return {"state": state, "n_fills": len(fill_tids), "n_twap": len(twap_all),
            "overlap": len(overlap), "n_twap_in_window": n_twap_win, "n_fills_in_window": n_fill_win,
            "window": [t_lo, t_hi], "n_zero_hash": len(zero),
            "zero_hash_by_dir": dict(Counter(str(f.get("dir")) for f in zero)),
            "zero_hash_crossed": dict(Counter(str(f.get("crossed")) for f in zero))}


def wallet_record(w, samp, fills, pm, twap_rows, end_ms, tol_ms, mixed_run, run_id):
    """Build one wallet's checkpoint record, including the overlap check. Also used by --recheck."""
    times = [fill_time(f) for f in fills if fill_time(f)]
    t_first = min(times) if times else None
    t_last = max(times) if times else None
    twap_first = min((int(r["time"]) for r in twap_rows), default=None)
    t_lo = max(t_first or 0, twap_first or 0) + tol_ms
    overlap = check_source_overlap(fills, twap_rows, t_lo, end_ms)
    twap_tid = {int(r["tid"]): int(r["twap_id"]) for r in twap_rows if r.get("tid") is not None}
    rows = []
    for f in fills:
        tid = fill_tid(f)
        if tid is None:
            continue
        rows.append({"wallet": w, "tid": tid, "time": fill_time(f), "coin": f.get("coin"),
                     "side": f.get("side"), "px": f.get("px"), "sz": f.get("sz"),
                     "zero_hash": str(f.get("hash", "")).startswith(ZERO),
                     "is_twap": tid in twap_tid, "twap_id": twap_tid.get(tid),
                     "oid": f.get("oid"), "dir": f.get("dir"), "crossed": f.get("crossed"),
                     "sampling": samp, "run_id": run_id, "mixed_run": mixed_run})
    return {"wallet": w, "sampling": samp, "state": "ok", "mixed_run": mixed_run, "run_id": run_id,
            "n_fills": len(rows), "t_first_ms": t_first, "t_last_ms": t_last,
            "n_twap_labeled": sum(1 for r in rows if r["is_twap"]),
            "twap_earliest_ms": twap_first, "window_start_ms": t_lo,
            "paging": pm, "source_overlap": overlap, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Collect each sampled wallet's non-TWAP fills (userFillsByTime) for the mixed stream")
    ap.add_argument("--resume", action="store_true", help="resume from the checkpoint (finished wallets are not refetched)")
    # The two heaviest targeted wallets (128k and 186k fills) need --max-pages 100
    ap.add_argument("--max-pages", type=int, default=40, help="per-wallet page cap (40 pages = 80,000 fills)")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--tol", type=float, default=600, help="start margin of each wallet's check window (seconds)")
    ap.add_argument("--recheck", action="store_true", help="offline: recompute the overlap check and the report from the checkpoint")
    args = ap.parse_args()
    if args.recheck:
        return recheck(args)

    run_id = json.loads((OUT / "collect_report.json").read_text())["run_id"]
    ck = Checkpoint(run_id, CKPT, sub="fills")
    plan = Checkpoint(run_id, CKPT).plan()      # Wallets and window come from the TWAP run
    wallets = plan["picked"] + plan["targeted"]
    samp = {w: "random" for w in plan["picked"]}
    samp.update({w: "targeted" for w in plan["targeted"]})
    end_ms = plan["now_ms"]     # Not later: TWAPs started after this aren't labeled, they'd count as noise
    ck.wdir.mkdir(parents=True, exist_ok=True)

    twap_by_w = defaultdict(list)
    for line in (OUT / "slices.jsonl").open():
        if line.strip():
            r = json.loads(line)
            twap_by_w[r["wallet"]].append(r)

    mixed_run = f"MIX-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    print(f"mixed run {mixed_run}, TWAP run {run_id}, window end {end_ms}, wallets {len(wallets)}"
          + (" (resumed)" if args.resume else ""))
    client = default_client()
    client.timeout = args.timeout
    print(f"API ok ({client.preflight()} s)")
    fetch = make_fills_fetch(client)

    t0 = time.monotonic()
    streak = 0
    failed = []
    for i, w in enumerate(wallets, 1):
        prev = ck.load_wallet(w)
        if prev and prev.get("state") == "ok":
            print(f"  [{i}/{len(wallets)}] {w[:10]}... already collected")
            continue
        try:
            fills, pm = paginate_adaptive(fetch, w, 0, end_ms, max_pages=args.max_pages)
            if pm["hit_max_pages"]:
                raise APIRequestFailed(f"hit the page cap {args.max_pages} ({len(fills):,} fills fetched)")
            if pm["dedup_suspicious"]:
                raise APIRequestFailed(f"suspicious tid de-duplication ({pm['deduped']:,} dropped)")
        except APIRequestFailed as e:
            streak += 1
            ck.save_wallet(w, {"wallet": w, "sampling": samp[w], "state": "failed",
                               "error": str(e)[:240], "mixed_run": mixed_run})
            failed.append({"wallet": w, "error": str(e)[:160]})
            print(f"  [{i}/{len(wallets)}] {w[:10]}... FAIL ({str(e)[:90]})")
            if streak >= BREAKER:
                sys.exit(f"{BREAKER} wallets in a row failed, stopping (--resume to continue)")
            continue
        streak = 0
        rec = wallet_record(w, samp[w], fills, pm, twap_by_w.get(w, []), end_ms,
                            int(args.tol * 1000), mixed_run, run_id)
        ck.save_wallet(w, rec)
        so = rec["source_overlap"]
        el = time.monotonic() - t0
        print(f"  [{i}/{len(wallets)}] {w[:10]}... {samp[w]:<8} {rec['n_fills']:>6} fills, "
              f"in window TWAP {so['n_twap_in_window']:>6} / non-TWAP {so['n_fills_in_window']:>6}, "
              f"overlap {so['state']}, {el/60:.1f} min (waited {client.stats['waited_s']/60:.1f})", flush=True)

    return finalize(ck, wallets, mixed_run, run_id, end_ms, failed, client.stats)


def finalize(ck, wallets, mixed_run, run_id, end_ms, failed, client_stats):
    states = ck.states(wallets)
    pending = [w for w, s in states.items() if s is None]
    if pending:
        sys.exit(f"{len(pending)} wallets not collected yet")
    # mixed_run is this call's id. Wallets fetched by an earlier call keep theirs in the rows,
    # the report only has the last one.
    all_rows, detail = [], []
    zh_dir, zh_cross = Counter(), Counter()
    for w in wallets:
        rec = ck.load_wallet(w) or {}
        if rec.get("state") != "ok":
            detail.append({"wallet": w, "state": "failed", "error": rec.get("error")})
            continue
        all_rows += rec["rows"]
        detail.append({k: rec.get(k) for k in ("wallet", "sampling", "state", "n_fills", "t_first_ms", "t_last_ms",
                                                "n_twap_labeled", "twap_earliest_ms", "window_start_ms", "source_overlap")})
        zh_dir.update(rec["source_overlap"].get("zero_hash_by_dir", {}))
        zh_cross.update(rec["source_overlap"].get("zero_hash_crossed", {}))
    atomic_jsonl(OUT / "mixed_fills.jsonl", all_rows)
    overlap_states = Counter(d["source_overlap"]["state"] for d in detail if d["state"] == "ok")
    # The server may keep only the latest ~10,000 fills per wallet, count who might be cut off
    n_cap = sum(1 for d in detail if d["state"] == "ok" and d["n_fills"] >= 10_000)
    n_short = sum(1 for d in detail if d["state"] == "ok" and d.get("twap_earliest_ms")
                  and d["t_first_ms"] and d["t_first_ms"] > d["twap_earliest_ms"])
    status = "ok" if not failed else "degraded"
    out = {"mixed_run": mixed_run, "run_id": run_id, "status": status, "end_ms": end_ms,
           "design": "union of userFills (non-TWAP, singletons) and slices (TWAP, metaorders); "
                     "the source overlap check passes when the two sources share no tid and both have fills inside the window",
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_wallets_ok": sum(1 for d in detail if d["state"] == "ok"), "failed": failed,
           "n_fills": len(all_rows), "n_twap_labeled": sum(1 for r in all_rows if r["is_twap"]),
           "n_zero_hash": sum(1 for r in all_rows if r["zero_hash"]),
           "source_overlap_states": dict(overlap_states), "zero_hash_by_dir": dict(zh_dir), "zero_hash_crossed": dict(zh_cross),
           "n_wallets_ge_10k": n_cap, "n_wallets_fills_history_shorter_than_twap": n_short,
           "client": client_stats, "wallets": detail}
    (OUT / "mixed_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nuserFills {len(all_rows):,} fills, TWAP tids among them {out['n_twap_labeled']:,} (should be 0), "
          f"zero-hash {out['n_zero_hash']:,} (by dir {dict(zh_dir)})")
    print(f"overlap check {dict(overlap_states)}, wallets with >=10,000 fills {n_cap}, "
          f"userFills history shorter than TWAP history {n_short}, status {status}")
    print(f"wrote data/mixed_fills.jsonl, data/mixed_report.json ({mixed_run})")
    return 0


def recheck(args):
    """Offline: recompute each wallet's record and the report from checkpointed userFills."""
    run_id = json.loads((OUT / "collect_report.json").read_text())["run_id"]
    ck = Checkpoint(run_id, CKPT, sub="fills")
    plan = Checkpoint(run_id, CKPT).plan()
    wallets = plan["picked"] + plan["targeted"]
    samp = {w: "random" for w in plan["picked"]}
    samp.update({w: "targeted" for w in plan["targeted"]})
    end_ms = plan["now_ms"]
    twap_by_w = defaultdict(list)
    for line in (OUT / "slices.jsonl").open():
        if line.strip():
            r = json.loads(line)
            twap_by_w[r["wallet"]].append(r)
    old = json.loads((OUT / "mixed_report.json").read_text()) if (OUT / "mixed_report.json").exists() else {}
    mixed_run = old.get("mixed_run") or "MIX-unknown"
    n = 0
    for w in wallets:
        rec = ck.load_wallet(w)
        if not rec or rec.get("state") != "ok":
            continue
        # rows keep only the zero_hash flag -> fake a hash that gives the same flag
        raw = [{"tid": r["tid"], "time": r["time"], "coin": r["coin"], "side": r["side"], "px": r["px"],
                "sz": r["sz"], "hash": ZERO if r.get("zero_hash") else "0x1", "oid": r.get("oid"),
                "dir": r.get("dir"), "crossed": r.get("crossed")} for r in rec["rows"]]
        new = wallet_record(w, samp[w], raw, rec.get("paging", {}), twap_by_w.get(w, []), end_ms,
                            int(args.tol * 1000), rec.get("mixed_run", mixed_run), run_id)
        ck.save_wallet(w, new)
        n += 1
    print(f"recheck: {n} wallets recomputed")
    return finalize(ck, wallets, mixed_run, run_id, end_ms, old.get("failed", []), old.get("client"))


if __name__ == "__main__":
    sys.exit(main())
