#!/usr/bin/env python3
"""
Collect the labeled metaorder dataset: each sampled wallet's `userTwapSliceFillsByTime`
fills, grouped by the exchange-assigned `twapId`. Each group is one ground-truth metaorder.
Wallets with sampling "targeted" were added by hand, so they don't stand for the pool.
metaorders.jsonl, slices.jsonl and collect_report.json are written only after both layers finish.

Usage:
    python3 scripts/collect_metaorders.py --wallets 60 --include-top 20
    python3 scripts/collect_metaorders.py --wallets 60 --include-top 20 --resume
    python3 scripts/collect_metaorders.py --wallets 60 --include-top 20 --resume --layer targeted
    python3 scripts/collect_metaorders.py --offline-report

--resume skips wallets already collected. Pass the same sampling flags as the run being resumed.
"""

import argparse
import json
import random
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
CKPT = OUT / "checkpoints"

from glassbook.checkpoint import Checkpoint, atomic_jsonl
from glassbook.hl.client import APIRequestFailed, default_client
from glassbook.hl.paging import (DAY_MS, DEFAULT_MAX_PAGES, HOUR_MS, fill_tid, fill_time,
                                 paginate, paginate_adaptive)
from glassbook.hl.stream import analyze_stream

SEED = 20260823
BREAKER = 3          # Consecutive failed wallets before giving up, counted per layer


def make_fetch():
    c = default_client()

    def fetch(user, start_ms, end_ms):
        r = c.post({"type": "userTwapSliceFillsByTime", "user": user,
                    "startTime": int(start_ms), "endTime": int(end_ms)})
        if not isinstance(r, list):
            raise APIRequestFailed(f"non-list response: {str(r)[:100]}")
        return r
    return fetch


def window_overlap(single: list, start_ms: int, end_ms: int) -> tuple[int, int] | None:
    if not single:
        return None
    s_lo, s_hi = min(fill_time(x) for x in single), max(fill_time(x) for x in single)
    lo, hi = max(s_lo, start_ms), min(s_hi, end_ms)
    return (lo, hi) if lo <= hi else None


def check_pagination(fetch, post, user: str, start_ms: int, end_ms: int) -> dict:
    """Pagination completeness check ("pagination_check" in the report).

    Compares the paged result with one unpaginated request over the same window, and with the
    userTwapSliceFills tids that fall inside the window. A tid missing from the paged result is a
    fail, a request that errored makes the whole check an error.
    """
    # failed request -> "error", never []: the empty set is a subset of anything, [] would pass
    errors = []
    try:
        paged, meta = paginate(fetch, user, start_ms, end_ms)
    except APIRequestFailed as e:
        return {"wallet": user, "state": "error", "errors": [f"pagination: {e}"]}
    try:
        one = fetch(user, start_ms, end_ms)
    except APIRequestFailed as e:
        errors.append(f"single call, same window: {e}")
        one = None
    try:
        single = post({"type": "userTwapSliceFills", "user": user})
        if not isinstance(single, list):
            raise APIRequestFailed(f"non-list response: {str(single)[:80]}")
    except APIRequestFailed as e:
        errors.append(f"userTwapSliceFills: {e}")
        single = None

    if errors:
        return {"wallet": user, "state": "error", "errors": errors, "meta": meta,
                "paged": len(paged)}

    pt = {fill_tid(x) for x in paged} - {None}
    ot = {fill_tid(x) for x in one} - {None}
    vs_single = "pass" if ot <= pt else "fail"
    ov = window_overlap(single, start_ms, end_ms)
    if ov is None:
        vs_unpaged = {"state": "inconclusive",
               "reason": "userTwapSliceFills has no fills inside the window" if single else "userTwapSliceFills is empty for this wallet"}
    else:
        lo, hi = ov
        s_in = {fill_tid(x) for x in single if lo <= fill_time(x) <= hi} - {None}
        p_in = {fill_tid(x) for x in paged if lo <= fill_time(x) <= hi} - {None}
        vs_unpaged = {"state": "pass" if s_in <= p_in else "fail", "overlap_ms": [lo, hi],
               "single_in_overlap": len(s_in), "paged_in_overlap": len(p_in),
               "missing": len(s_in - p_in)}

    if vs_single == "fail" or vs_unpaged["state"] == "fail":
        state = "fail"
    elif vs_unpaged["state"] == "inconclusive":
        state = "inconclusive"
    else:
        state = "pass"
    return {"wallet": user, "state": state, "paged": len(pt), "single_call": len(ot),
            "twapslicefills": len(single), "meta": meta,
            "vs_single_request": {"state": vs_single, "missing": len(ot - pt)}, "vs_unpaged_endpoint": vs_unpaged}


def build_pool():
    files = sorted(RAW.glob("trades_*.jsonl"))
    if not files:
        sys.exit("no trades_*.jsonl in data/raw/; run check_feasibility.py first")
    freq = defaultdict(int)
    total = 0
    for fp in files:
        s = analyze_stream(fp)
        total += s["trades_total"]
        # Inferred takers only. The other side of a zero-hash trade is whoever was resting there.
        aggr = set(s.get("aggressor_set") or [])
        for w, n in s["twap_wallet_ranking"]:
            if w in aggr:
                freq[w] += n
    counts = sorted(freq.values(), reverse=True)
    return sorted(freq), {
        "stream_files": len(files),
        "trades_scanned": total,
        "pool_size": len(freq),
        "appearances_median": statistics.median(counts) if counts else 0,
        "appearances_p90": counts[int(.1 * len(counts))] if counts else 0,
        "appearances_max": counts[0] if counts else 0,
        "top10pct_share": round(sum(counts[:max(1, len(counts) // 10)]) / sum(counts), 3)
        if counts else 0,
    }


def assemble(wallet: str, fills: list, window_start_ms: int, window_end_ms: int,
             tol_s: float = 600.0) -> tuple[list[dict], list[dict]]:
    """Group fills by twapId into metaorders and flag truncation at either end.

    One TWAP slice can sweep several price levels, so the per-metaorder count is `fills`.
    Metaorders starting within `tol_s` of the left edge are truncated_left, those ending
    within `tol_s` of window_end_ms may still be running (truncated_right).
    """
    groups: defaultdict[int, list[dict]] = defaultdict(list)
    for item in fills:
        if not isinstance(item, dict):
            continue
        f = item.get("fill", item)
        tid = item.get("twapId", f.get("twapId"))
        if tid is not None:
            groups[int(tid)].append(f)

    # "full history" isn't: one wallet gave 6,191 fills, 17 h later 3,711 (retention moves).
    # left edge = earliest fill we got for the wallet, not the window start
    all_times = [fill_time(f) for fs in groups.values() for f in fs if fill_time(f)]
    wallet_first = min(all_times) if all_times else 0
    left_edge = max(window_start_ms, wallet_first)

    metas, rows = [], []
    for twap_id, fs in groups.items():
        fs.sort(key=lambda f: (fill_time(f), f.get("tid") or 0))
        times = [fill_time(f) for f in fs if fill_time(f)]
        if not times:
            continue
        szs, notionals = [], []
        for f in fs:
            try:
                px, sz = float(f.get("px", 0)), float(f.get("sz", 0))
            except (TypeError, ValueError):
                px = sz = 0.0
            szs.append(sz)
            notionals.append(px * sz)
            rows.append({"wallet": wallet, "twap_id": twap_id, "tid": f.get("tid"),
                         "time": fill_time(f), "coin": f.get("coin"), "side": f.get("side"),
                         "px": f.get("px"), "sz": f.get("sz")})
        t0, t1 = min(times), max(times)
        left = left_edge > 0 and t0 <= left_edge + tol_s * 1000
        right = t1 >= window_end_ms - tol_s * 1000
        metas.append({"wallet": wallet, "twap_id": twap_id, "coin": fs[0].get("coin"),
                      "side": fs[0].get("side"), "fills": len(fs),
                      "size": round(sum(szs), 8), "notional_usd": round(sum(notionals), 2),
                      "t_start_ms": t0, "t_end_ms": t1,
                      "duration_s": round((t1 - t0) / 1000, 1),
                      "truncated_left": left, "truncated_right": right,
                      "is_boundary": left or right, "left_edge_ms": left_edge})
    return metas, rows


def rebuild_from_checkpoint(ckpt, tol_s):
    """Offline: re-run `assemble` on checkpointed fills, rewrite outputs and report summaries."""
    plan = ckpt.plan()
    rp = OUT / "collect_report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    start, now = plan["start_ms"], plan["now_ms"]
    all_metas, all_rows, n_wallets, changed = [], [], 0, 0
    for w in plan["picked"] + plan["targeted"]:
        rec = ckpt.load_wallet(w)
        if not rec or rec.get("state") != "ok":
            continue
        items = [{"fill": r, "twapId": r["twap_id"]} for r in rec["rows"]]
        ms, rs = assemble(w, items, start, now, tol_s=tol_s)
        old_left = sum(1 for m in rec["metas"] if m.get("truncated_left"))
        for x in ms:
            x["sampling"] = rec["sampling"]
            x["run_id"] = plan["run_id"]
        for x in rs:
            x["sampling"] = rec["sampling"]
            x["run_id"] = plan["run_id"]
        # rebuild: flags only, row counts must not change
        assert len(ms) == len(rec["metas"]) and len(rs) == len(rec["rows"]), w
        changed += sum(1 for m in ms if m["truncated_left"]) - old_left
        rec["metas"] = ms
        ckpt.save_wallet(w, rec)
        all_metas += ms
        all_rows += rs
        n_wallets += 1
    atomic_jsonl(OUT / "metaorders.jsonl", all_metas)
    atomic_jsonl(OUT / "slices.jsonl", all_rows)
    rand_metas = [x for x in all_metas if x["sampling"] == "random"]
    rep["summary_random"] = summarize(rand_metas)
    rep["summary_targeted"] = summarize([x for x in all_metas if x["sampling"] == "targeted"])
    rep.setdefault("rebuilt", []).append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": "re-assembled from checkpoint",
        "wallets": n_wallets, "truncated_left_delta": changed})
    write_report(rep)
    return {"wallets": n_wallets, "metaorders": len(all_metas), "rows": len(all_rows),
            "truncated_left_delta": changed}


def summarize(metas):
    def stats(rows):
        if not rows:
            return {"n": 0}
        fl = sorted(m["fills"] for m in rows)
        no = sorted(m["notional_usd"] for m in rows if m["notional_usd"] > 0)
        return {"n": len(rows),
                "fills_median": statistics.median(fl),
                "fills_mean": round(statistics.mean(fl), 1),
                "fills_p90": fl[min(len(fl) - 1, int(.9 * len(fl)))],
                "notional_median": round(statistics.median(no), 2) if no else 0,
                "notional_p90": round(no[min(len(no) - 1, int(.9 * len(no)))], 2) if no else 0,
                "n_ge_1m": sum(1 for x in no if x >= 1_000_000),
                "wallets": len({m["wallet"] for m in rows}),
                "coins": len({m["coin"] for m in rows})}
    clean = [m for m in metas if not m["is_boundary"]]
    return {"all": stats(metas), "boundary_excluded": stats(clean),
            "boundary_dropped": len(metas) - len(clean),
            "truncated_left": sum(1 for m in metas if m.get("truncated_left")),
            "truncated_right": sum(1 for m in metas if m.get("truncated_right"))}


def collect_layer(fetch, ckpt, wallets, tag,
                  start_ms, end_ms, run_id, tol_s=600.0,
                  max_pages=DEFAULT_MAX_PAGES):
    """Collect one layer, saving each wallet to the checkpoint as soon as it's done, so a
    breaker trip in the targeted layer doesn't cost the random one."""
    streak, n_ok, n_skip, failed = 0, 0, 0, []
    t_start = time.monotonic()
    for i, w in enumerate(wallets, 1):
        prev = ckpt.load_wallet(w)
        if prev and prev.get("state") == "ok":
            n_skip += 1
            print(f"  [{i}/{len(wallets)}] {w[:10]}... already collected")
            continue
        t0 = time.monotonic()
        try:
            fills, pm = paginate_adaptive(fetch, w, start_ms, end_ms, max_pages=max_pages)
            if pm["hit_max_pages"]:
                raise APIRequestFailed(f"hit the page cap ({max_pages} pages, {len(fills):,} fills)")
            if pm["dedup_suspicious"]:
                raise APIRequestFailed(f"suspicious tid de-duplication ({pm['deduped']:,} dropped)")
        except APIRequestFailed as e:
            streak += 1
            rec = {"wallet": w, "sampling": tag, "state": "failed", "run_id": run_id,
                   "error": str(e)[:240], "elapsed_s": round(time.monotonic() - t0, 1)}
            ckpt.save_wallet(w, rec)
            failed.append({"wallet": w, "sampling": tag, "error": rec["error"]})
            print(f"  [{i}/{len(wallets)}] {w[:10]}... FAIL ({str(e)[:90]})")
            if streak >= BREAKER:
                return {"ok": n_ok, "skipped": n_skip, "failed": failed, "tripped": True}
            continue
        streak = 0
        ms, rs = assemble(w, fills, start_ms, end_ms, tol_s=tol_s)
        for x in ms + rs:
            x["sampling"] = tag
            x["run_id"] = run_id
        times = [r["time"] for r in rs if r["time"]]
        rec = {"wallet": w, "sampling": tag, "state": "ok", "run_id": run_id,
               "fills": len(fills), "metaorders": len(ms), "paging": pm,
               "t_first_ms": min(times) if times else None,
               "t_last_ms": max(times) if times else None,
               "elapsed_s": round(time.monotonic() - t0, 1), "metas": ms, "rows": rs}
        ckpt.save_wallet(w, rec)
        n_ok += 1
        extra = f", {pm['windows']} windows ({pm['failures']} failures)" if pm["splits"] else ""
        print(f"  [{i}/{len(wallets)}] {w[:10]}... {len(fills):>6} fills / {len(ms):>3} metaorders"
              f", {(time.monotonic() - t_start) / 60:.1f} min"
              f" (waited {default_client().stats['waited_s'] / 60:.1f}){extra}")
    return {"ok": n_ok, "skipped": n_skip, "failed": failed, "tripped": False}


def write_report(obj):
    (OUT / "collect_report.json").write_text(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Collect the labeled metaorder dataset (TWAP fills grouped by twapId)")
    ap.add_argument("--wallets", type=int, default=60, help="number of wallets sampled uniformly at random")
    ap.add_argument("--days", type=int, default=0, help="look-back in days; 0 = full history (default, startTime=0)")
    ap.add_argument("--pagination-tries", type=int, default=6,
                    help="wallets to try for the pagination check before giving up")
    ap.add_argument("--include-top", type=int, default=0,
                    help="also collect the top N TWAP wallets from feasibility_report.json (tagged targeted)")
    ap.add_argument("--include", type=str, default="", help="comma-separated addresses to add as targeted")
    ap.add_argument("--boundary-tol", type=float, default=600)
    ap.add_argument("--contamination", type=int, default=15,
                    help="number of pool wallets outside the sample probed in the contamination check")
    ap.add_argument("--layer", choices=("all", "random", "targeted"), default="all",
                    help="run one layer only, the other one can follow with --resume")
    ap.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
                    help="resume the latest run (or RUN_ID) from its checkpoint")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help="per-wallet page cap; a wallet that hits it is marked failed")
    ap.add_argument("--timeout", type=float, default=45.0, help="per-request read timeout (seconds)")
    ap.add_argument("--offline-report", action="store_true")
    ap.add_argument("--rebuild", nargs="?", const="latest", default=None, metavar="RUN_ID",
                    help="offline: re-assemble from the checkpoint and rewrite metaorders/slices (e.g. with a different --boundary-tol)")
    args = ap.parse_args()

    if args.rebuild:
        rid = Checkpoint.latest(CKPT) if args.rebuild == "latest" else args.rebuild
        if not rid or not Checkpoint(rid, CKPT).exists():
            sys.exit(f"no checkpoint {rid}")
        r = rebuild_from_checkpoint(Checkpoint(rid, CKPT), args.boundary_tol)
        print(f"rebuilt from checkpoint {rid}: {r['wallets']} wallets, {r['metaorders']} metaorders, "
              f"{r['rows']} fills, truncated_left change {r['truncated_left_delta']:+d}")
        return

    if args.offline_report:
        metas = [json.loads(l) for l in (OUT / "metaorders.jsonl").open() if l.strip()]
        print(json.dumps(summarize([m for m in metas if m.get("sampling") == "random"]), indent=2))
        return

    client = default_client()
    client.timeout = args.timeout

    if args.resume:
        rid = Checkpoint.latest(CKPT) if args.resume == "latest" else args.resume
        if not rid or not Checkpoint(rid, CKPT).exists():
            sys.exit(f"no checkpoint {rid}")
        ckpt = Checkpoint(rid, CKPT)
        plan = ckpt.plan()
        # The wallets come from the checkpoint anyway, but boundary_tol goes into the flags
        for k in ("wallets", "days", "include_top", "include", "boundary_tol"):
            if plan["params"].get(k) != getattr(args, k):
                sys.exit(f"--{k.replace('_', '-')} is {getattr(args, k)!r} here but {plan['params'].get(k)!r} in the checkpoint")
        run_id = plan["run_id"]
        profile, pool = plan["pool_profile"], plan["pool"]
        picked, targeted, pcheck = plan["picked"], plan["targeted"], plan["pagination_check"]
        now, start = plan["now_ms"], plan["start_ms"]
        report = {"run_id": run_id, "status": "running", "params": vars(args), "seed": SEED,
                  "pool_profile": profile, "pagination_attempts": plan["pagination_attempts"],
                  "resumed": True, "checkpoint": str(ckpt.dir)}
        st = ckpt.states(picked + targeted)
        print(f"resuming run {run_id}: {sum(1 for s in st.values() if s == 'ok')} done, "
              f"{sum(1 for s in st.values() if s == 'failed')} failed, "
              f"{sum(1 for s in st.values() if s is None)} not started")
        print(f"API ok ({client.preflight()} s)")
        fetch = make_fetch()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        report = {"run_id": run_id, "status": "running", "params": vars(args), "seed": SEED}
        pool, profile = build_pool()
        report["pool_profile"] = profile
        print(f"run {run_id}")
        print(f"wallet pool {profile['pool_size']:,} ({profile['stream_files']} stream files, "
              f"{profile['trades_scanned']:,} trades), top 10% = {profile['top10pct_share']:.1%} of appearances")
        print(f"API ok ({client.preflight()} s)")

        random.seed(SEED)
        picked = random.sample(pool, min(args.wallets, len(pool)))
        rand_set = set(picked)
        targeted = []
        if args.include:
            targeted += [w.strip().lower() for w in args.include.split(",") if w.strip()]
        if args.include_top:
            top = json.loads((OUT / "feasibility_report.json").read_text())["stream"]["twap_wallet_top20"]
            targeted += [w for w, _ in top[:args.include_top]]
        targeted = [w for w in dict.fromkeys(targeted) if w not in rand_set]
        print(f"  random {len(picked)} + targeted {len(targeted)}")

        now = int(time.time() * 1000)
        start = 0 if args.days <= 0 else now - args.days * DAY_MS
        fetch = make_fetch()

        print("\nPagination check")
        pcheck, tried = None, []
        for i, w in enumerate(picked[:args.pagination_tries], 1):
            r = check_pagination(fetch, client.post, w, start, now)
            tried.append(r)
            if r["state"] == "error":
                print(f"  [{i}] {w[:10]}... error: {'; '.join(r['errors'])[:140]}")
                if sum(1 for x in tried if x["state"] == "error") >= BREAKER:
                    break
                continue
            if r["state"] == "inconclusive":
                print(f"  [{i}] {w[:10]}... inconclusive ({r['vs_unpaged_endpoint']['reason']}), next wallet")
                continue
            print(f"  [{i}] {w[:10]}... {r['state']}: paged {r['paged']:,}, single call {r['single_call']:,}, "
                  f"missing from paged {r['vs_unpaged_endpoint'].get('missing', 'n/a')}")
            pcheck = r
            break
        report["pagination_attempts"] = tried
        if pcheck is None or pcheck["state"] != "pass":
            sys.exit("pagination check didn't pass, nothing written")
        if pcheck["meta"]["dedup_suspicious"] or pcheck["meta"]["hit_max_pages"]:
            sys.exit("pagination check: suspicious de-duplication or page cap hit, nothing written")

        ckpt = Checkpoint(run_id, CKPT)
        ckpt.create({"run_id": run_id, "created": datetime.now(timezone.utc).isoformat(),
                     "params": vars(args), "seed": SEED, "now_ms": now, "start_ms": start,
                     "pool_profile": profile, "pool": pool, "picked": picked,
                     "targeted": targeted, "pagination_check": pcheck, "pagination_attempts": tried})
        report["checkpoint"] = str(ckpt.dir)

    rand_set = set(picked)
    layer_results = {}
    for tag, ws in (("random", picked), ("targeted", targeted)):
        if args.layer in ("all", tag) and ws:
            print(f"\n{tag} layer, {len(ws)} wallets (budget {int(client.budget)} weight/min)")
            r = collect_layer(fetch, ckpt, ws, tag, start, now, run_id,
                              tol_s=args.boundary_tol, max_pages=args.max_pages)
            layer_results[tag] = {k: v for k, v in r.items() if k != "failed"}
            if r["tripped"]:
                sys.exit(f"{BREAKER} wallets in a row failed in the {tag} layer, stopping. "
                         "Finished wallets are in the checkpoint, --resume picks up from there.")

    # The outputs cover both layers
    pending = [w for w, s in ckpt.states(picked + targeted).items() if s is None]
    if pending:
        other = "random" if any(w in rand_set for w in pending) else "targeted"
        sys.exit(f"{len(pending)} {other} wallets not collected yet, outputs not written (--resume --layer {other})")
    failed = []
    for w in picked + targeted:
        rec = ckpt.load_wallet(w) or {}
        if rec.get("state") == "failed":
            failed.append({"wallet": w, "sampling": rec.get("sampling"), "error": rec.get("error")})

    # Contamination check: do pool wallets outside the sample have TWAP fills too?
    print(f"\nContamination check, {args.contamination} pool wallets outside the sample")
    rest = [w for w in pool if w not in set(picked + targeted)]
    random.seed(SEED + 1)
    probe = random.sample(rest, min(args.contamination, len(rest)))
    c_hit = c_miss = c_err = 0
    for w in probe:
        try:
            r = client.post({"type": "userTwapSliceFills", "user": w})
        except APIRequestFailed as e:
            c_err += 1      # neither a hit nor a miss
            continue
        if isinstance(r, list) and r:
            c_hit += 1
        else:
            c_miss += 1
    c_n = c_hit + c_miss
    c_rate = round(c_hit / c_n, 3) if c_n else None
    print(f"  {c_hit}/{c_n} have TWAP fills ({c_err} requests failed)")

    all_metas, all_rows, detail = [], [], []
    for w in picked + targeted:
        rec = ckpt.load_wallet(w) or {}
        if rec.get("state") != "ok":
            continue
        all_metas += rec["metas"]
        all_rows += rec["rows"]
        pg = rec.get("paging") or {}
        detail.append({"wallet": w, "sampling": rec["sampling"], "fills": rec["fills"],
                       "metaorders": rec["metaorders"], "pages": pg.get("pages"),
                       "windows": pg.get("windows"), "splits": pg.get("splits"),
                       "t_first_ms": rec.get("t_first_ms"), "t_last_ms": rec.get("t_last_ms")})
    atomic_jsonl(OUT / "metaorders.jsonl", all_metas)
    atomic_jsonl(OUT / "slices.jsonl", all_rows)

    rand_metas = [x for x in all_metas if x["sampling"] == "random"]
    targ_failed = [f for f in failed if f["sampling"] == "targeted"]
    status = "ok" if not failed else ("degraded_targeted" if targ_failed else "degraded")
    summ = summarize(rand_metas)
    write_report({**report, "status": status,
                  "pagination_check": pcheck, "client": client.stats, "layers": layer_results,
                  "window": {"start_ms": start, "end_ms": now},
                  "wallets": {"random": len(picked), "targeted": len(targeted),
                              "failed": failed, "detail": detail},
                  "contamination": {"hit_rate": c_rate, "n_valid": c_n, "n_error": c_err},
                  "summary_random": summ,
                  "summary_targeted": summarize([x for x in all_metas
                                                 if x["sampling"] == "targeted"])})
    ckpt.update_plan(completed=True, completed_at=datetime.now(timezone.utc).isoformat(),
                     status=status)

    a = summ["all"]
    print(f"\nrandom layer {a.get('n', 0)} metaorders / {a.get('wallets', 0)} wallets, "
          f"targeted {len(all_metas) - len(rand_metas)} metaorders, failed wallets {len(failed)} ({status})")
    print(f"API {client.stats['calls']} calls, weight {client.stats['weight']:,}, "
          f"waited {client.stats['waited_s'] / 60:.1f} min")
    print(f"wrote data/metaorders.jsonl, data/slices.jsonl, data/collect_report.json (run {run_id})")


if __name__ == "__main__":
    main()
