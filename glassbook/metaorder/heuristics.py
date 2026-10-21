"""Metaorder reconstruction heuristics for an anonymous tape, and scoring against exchange labels.

Heuristics see only public tape fields (wallet, coin, side, time, size, taker flag), never oid, cloid
or twapId. Those are the labels.

Metrics (t = scored metaorder, c = predicted cluster, n_tc = fills of t in c):
  pair_f1         all fill pairs.
  pair_f1_focus   headline, only pairs touching a scored metaorder. TP = sum C(n_tc, 2),
                  predicted = sum_c [C(n_c, 2) - C(m_c, 2)] with m_c = fills in c outside every scored
                  metaorder, true = sum_t C(n_t, 2).
  jaccard_focus   F1 / (2 - F1).
  merge_twap      a cluster holding t also holds fills of another TWAP metaorder.
  contam_nontwap  a cluster holding t holds at least one non-TWAP fill. merge_rate is the union of the two.
  notional ratio  N_hat_t / N_t with N_hat_t = sum_c (N_tc / N_t) * N(c). Reported as median, p90, geomean,
                  mean, share > 2, share > 10, and aggregate sum(N_hat) / sum(N). count_ratio is the
                  same ratio by fill count instead of notional.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Hashable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Fill:
    """fid must be unique across accounts. Both counterparties report the same tid, so key on (wallet, tid)."""
    fid: int
    wallet: str
    coin: str
    side: str          # "B" = buy, "A" = sell
    ts_ms: int
    px: float
    sz: float
    crossed: bool | None = None   # Taker flag, always True for TWAP slices

    @property
    def notional(self) -> float:
        return self.px * self.sz


Clustering = dict[int, Hashable]   # fid -> group id


def sort_fills(fills: Sequence[Fill]) -> list[Fill]:
    """Sort by (ts_ms, fid). Every entry point sorts first, input order can leak labels (rows grouped by twapId)."""
    return sorted(fills, key=lambda f: (f.ts_ms, f.fid))


def h1_gap_runs(fills: Sequence[Fill], tau_s: float = 300.0) -> Clustering:
    """H1: per (wallet, coin, side), a gap over tau_s starts a new metaorder."""
    out = {}
    by_key = defaultdict(list)
    for f in fills:
        by_key[(f.wallet, f.coin, f.side)].append(f)
    tau_ms = tau_s * 1000
    for key, group in by_key.items():
        group.sort(key=lambda f: (f.ts_ms, f.fid))
        run, prev_ts = 0, None
        for f in group:
            if prev_ts is not None and (f.ts_ms - prev_ts) > tau_ms:
                run += 1
            out[f.fid] = ("h1", *key, run)
            prev_ts = f.ts_ms
    return out


def h2_fixed_window(fills: Sequence[Fill], window_s: float = 1800.0) -> Clustering:
    """H2: per (wallet, coin, side) and fixed time bucket. Cuts metaorders that span a bucket edge."""
    # epoch-anchored buckets, not per wallet: this is the naive rule
    out = {}
    w_ms = window_s * 1000
    for f in fills:
        out[f.fid] = ("h2", f.wallet, f.coin, f.side, int(f.ts_ms // w_ms))
    return out


def h3_direction_runs(fills: Sequence[Fill], tau_s: float = 900.0) -> Clustering:
    """H3: per (wallet, coin), a side flip or a gap over tau_s starts a new metaorder."""
    out = {}
    by_key = defaultdict(list)
    for f in fills:
        by_key[(f.wallet, f.coin)].append(f)
    tau_ms = tau_s * 1000
    for key, group in by_key.items():
        group.sort(key=lambda f: (f.ts_ms, f.fid))
        run, prev_ts, prev_side = 0, None, None
        for f in group:
            if (prev_side is not None and f.side != prev_side) or (prev_ts is not None and (f.ts_ms - prev_ts) > tau_ms):
                run += 1
            out[f.fid] = ("h3", *key, run)
            prev_ts, prev_side = f.ts_ms, f.side
    return out


HEURISTICS: dict[str, Callable[..., Clustering]] = {
    "h1_gap_runs": h1_gap_runs,
    "h2_fixed_window": h2_fixed_window,
    "h3_direction_runs": h3_direction_runs,
}


def _c2(n: int) -> int:
    return n * (n - 1) // 2


def quantile(xs: Sequence[float], q: float) -> float | None:
    """Linear-interpolation quantile (numpy default). None for empty input."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _pair_counts(a: Clustering, b: Clustering, fids: Iterable[int]) -> tuple[int, int, int]:
    """Co-clustered pair counts (in both, only in a, only in b) from a contingency table, O(N)."""
    n_a = defaultdict(int)
    n_b = defaultdict(int)
    n_ab = defaultdict(int)
    for f in fids:
        ka, kb = a[f], b[f]
        n_a[ka] += 1
        n_b[kb] += 1
        n_ab[(ka, kb)] += 1
    inter = sum(_c2(n) for n in n_ab.values())
    return inter, sum(_c2(n) for n in n_a.values()) - inter, sum(_c2(n) for n in n_b.values()) - inter


def _pair_counts_bruteforce(a: Clustering, b: Clustering, fids: Iterable[int]) -> tuple[int, int, int]:
    """O(N^2) enumeration, only for cross-checking _pair_counts in tests."""
    fids = list(fids)
    ga = defaultdict(list)
    gb = defaultdict(list)
    for f in fids:
        ga[a[f]].append(f)
        gb[b[f]].append(f)

    def pairs(g):
        s = set()
        for members in g.values():
            if len(members) > 1:
                s.update(combinations(sorted(members), 2))
        return s
    pa, pb = pairs(ga), pairs(gb)
    inter = len(pa & pb)
    return inter, len(pa) - inter, len(pb) - inter


def _prf(tp: int, only_t: int, only_p: int) -> tuple[float | None, float | None, float | None]:
    if tp + only_t + only_p == 0:
        return None, None, None
    p = tp / (tp + only_p) if tp + only_p else 0.0
    r = tp / (tp + only_t) if tp + only_t else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def is_noise_key(k: Hashable) -> bool:
    return isinstance(k, tuple) and len(k) > 0 and k[0] == "noise"


def evaluate(truth: Clustering, pred: Clustering, fills: Sequence[Fill] | None = None,
             focus: Callable[[Hashable], bool] | None = None) -> dict:
    """Score pred against truth.

    truth: fid -> key. Non-TWAP fills have keys starting with "noise".
    pred: fid -> cluster id, must cover every fid in truth.
    fills: needed for the notional ratios.
    focus: truth key -> scored or not. Default is every non-noise key.
    """
    fids = sorted(truth)
    if not fids:
        return {"n_fills": 0, "note": "no ground-truth fills"}
    missing = [f for f in fids if f not in pred]
    if missing:
        raise ValueError(f"prediction does not cover {len(missing)} / {len(fids)} fills (it must cover every fid)"
                         f", e.g. fid {missing[0]}")
    focus = focus or (lambda k: not is_noise_key(k))

    inter, only_t, only_p = _pair_counts(truth, pred, fids)
    precision, recall, f1 = _prf(inter, only_t, only_p)

    t_groups = defaultdict(set)
    p_groups = defaultdict(set)
    for f in fids:
        t_groups[truth[f]].add(f)
        p_groups[pred[f]].add(f)
    focus_keys = {k for k in t_groups if focus(k)}
    noise_fids = {f for f in fids if is_noise_key(truth[f])}

    # Headline pairs: at least one fill in a scored metaorder
    n_tc = defaultdict(int)
    k_c = defaultdict(int)
    for f in fids:
        tk = truth[f]
        if tk in focus_keys:
            n_tc[(tk, pred[f])] += 1
            k_c[pred[f]] += 1
    tp_f = sum(_c2(n) for n in n_tc.values())
    true_f = sum(_c2(len(t_groups[k])) for k in focus_keys)
    pred_f = sum(_c2(len(m)) - _c2(len(m) - k_c.get(c, 0)) for c, m in p_groups.items())
    p_f, r_f, f1_f = _prf(tp_f, true_f - tp_f, pred_f - tp_f)
    jacc = f1_f / (2 - f1_f) if f1_f is not None else None

    notional = {f.fid: f.notional for f in fills} if fills is not None else None
    p_notional = {c: sum(notional[f] for f in m) for c, m in p_groups.items()} if notional else None

    # Per scored metaorder: exact = its fills are exactly one predicted cluster, nothing more
    exact = merged_twap = contam = merged_any = split = 0
    count_ratios = []
    notional_ratios = []
    sum_n = sum_nhat = 0.0
    p_sets = {frozenset(v) for v in p_groups.values()}
    for tk in focus_keys:
        members = t_groups[tk]
        clusters = {pred[f] for f in members}
        if frozenset(members) in p_sets:
            exact += 1
        if len(clusters) > 1:
            split += 1
        m_twap = any(truth[g] != tk and not is_noise_key(truth[g]) for c in clusters for g in p_groups[c])
        m_noise = any(g in noise_fids for c in clusters for g in p_groups[c])
        merged_twap += m_twap
        contam += m_noise
        merged_any += (m_twap or m_noise)
        count_ratios.append(sum(len(p_groups[pred[f]]) for f in members) / len(members) / len(members))
        if notional:
            # weight by where the notional sits. whole-cluster charge: a few stray fills in a 5k-fill
            # cluster blow up the mean
            n_t = sum(notional[f] for f in members)
            if n_t > 0:
                per_c = defaultdict(float)
                for f in members:
                    per_c[pred[f]] += notional[f]
                n_hat = sum(share / n_t * p_notional[c] for c, share in per_c.items())
                notional_ratios.append(n_hat / n_t)
                sum_n += n_t
                sum_nhat += n_hat
    n_t = len(focus_keys)

    def rate(x):
        return round(x / n_t, 4) if n_t else None

    def stats(rs: list[float]) -> dict:
        if not rs:
            return {"median": None, "p90": None, "geomean": None, "mean": None, "share_gt2": None, "share_gt10": None}
        return {"median": round(quantile(rs, 0.5), 3), "p90": round(quantile(rs, 0.9), 3),
                "geomean": round(math.exp(sum(math.log(r) for r in rs if r > 0) / len(rs)), 3),
                "mean": round(sum(rs) / len(rs), 3),
                "share_gt2": round(sum(1 for r in rs if r > 2) / len(rs), 4),
                "share_gt10": round(sum(1 for r in rs if r > 10) / len(rs), 4)}
    cr, nr = stats(count_ratios), stats(notional_ratios)
    return {
        "n_fills": len(fids), "n_scored_fills": sum(len(t_groups[k]) for k in focus_keys),
        "n_noise_fills": len(noise_fids),
        "n_truth_metaorders": n_t, "n_truth_groups_all": len(t_groups), "n_pred_metaorders": len(p_groups),
        # All pairs
        "pair_precision": None if precision is None else round(precision, 4),
        "pair_recall": None if recall is None else round(recall, 4),
        "pair_f1": None if f1 is None else round(f1, 4),
        # Headline: pairs touching a scored metaorder
        "pair_f1_focus": None if f1_f is None else round(f1_f, 4),
        "pair_precision_focus": None if p_f is None else round(p_f, 4),
        "pair_recall_focus": None if r_f is None else round(r_f, 4),
        "jaccard_focus": None if jacc is None else round(jacc, 4),
        "exact_recovered": exact, "exact_recovered_rate": rate(exact),
        "merge_rate": rate(merged_any), "merge_twap_rate": rate(merged_twap), "contam_nontwap_rate": rate(contam),
        "split_rate": rate(split),
        "count_ratio_median": cr["median"], "count_ratio_mean": cr["mean"], "count_ratio_p90": cr["p90"],
        "notional_ratio_median": nr["median"], "notional_ratio_p90": nr["p90"], "notional_ratio_geomean": nr["geomean"],
        "notional_ratio_mean": nr["mean"], "notional_ratio_share_gt2": nr["share_gt2"],
        "notional_ratio_share_gt10": nr["share_gt10"],
        "notional_ratio_aggregate": round(sum_nhat / sum_n, 4) if sum_n > 0 else None,
    }


def economic_truth(truth: Clustering, fills: Sequence[Fill]) -> Clustering:
    """Truth variant that merges same (wallet, coin, side) TWAPs whose spans overlap. Noise keys untouched.

    Parallel same-side TWAPs ~ one economic decision, for impact work gluing them is arguably right.
    """
    span = {}
    for f in fills:
        k = truth.get(f.fid)
        if k is None or is_noise_key(k):
            continue
        s = span.setdefault(k, [f.wallet, f.coin, f.side, f.ts_ms, f.ts_ms])
        s[3], s[4] = min(s[3], f.ts_ms), max(s[4], f.ts_ms)
    parent = {k: k for k in span}

    def find(x):     # Union-find with path halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by = defaultdict(list)
    for k, (w, c, sd, a, b) in span.items():
        by[(w, c, sd)].append((a, b, k))
    for rows in by.values():
        rows.sort()
        cur_end, cur_root = None, None
        for a, b, k in rows:
            if cur_end is not None and a <= cur_end:
                parent[find(k)] = find(cur_root)
                cur_end = max(cur_end, b)
            else:
                cur_end, cur_root = b, k
    return {fid: (("econ",) + tuple(find(k))[1:] if isinstance(find(k), tuple) else ("econ", find(k)))
            if not is_noise_key(k) else k
            for fid, k in truth.items()}


def sweep(truth: Clustering, fills: Sequence[Fill],
          taus: Sequence[float] = (60, 300, 900, 1800, 3600),
          focus: Callable[[Hashable], bool] | None = None,
          heuristics: dict[str, Callable[..., Clustering]] | None = None) -> list[dict]:
    """Evaluate every heuristic at every tau (window_s for h2). Fills are sorted first."""
    fills = sort_fills(fills)
    rows = []
    for name, fn in (heuristics or HEURISTICS).items():
        for tau in taus:
            kw = {"window_s": tau} if name == "h2_fixed_window" else {"tau_s": tau}
            m = evaluate(truth, fn(fills, **kw), fills=fills, focus=focus)
            rows.append({"heuristic": name, "tau_s": tau, **m})
    return rows


def truth_from_twap_ids(rows: Iterable[tuple[int, str, int]]) -> Clustering:
    """Truth from (fid, wallet, twap_id) rows. twapId is only unique per wallet, so the key includes the wallet."""
    return {fid: ("truth", wallet, twap_id) for fid, wallet, twap_id in rows}
