"""Impact of metaorders measured on public candles, and the power-law fit I = A * x^beta.

Used by estimate_impact.py and impact_power.py. Stdlib only.
"""

import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

from .candles import INTERVALS, candle_path

MIN_MS = 60_000
BAR_MS = dict(INTERVALS)
PERM_MS = 30 * MIN_MS
VOL_MS = 24 * 60 * MIN_MS
TIERS = [iv for iv, _ in INTERVALS]


class Candles:
    """Close and volume series per (coin, interval), loaded lazily from data/candles/."""

    def __init__(self, cdir):
        self.cdir = Path(cdir)
        self.ts: dict[tuple, list[int]] = {}
        self.close: dict[tuple, list[float]] = {}
        self.vol: dict[tuple, list[float]] = {}

    def load(self, coin, iv):
        key = (coin, iv)
        if key in self.ts:
            return bool(self.ts[key])
        p = candle_path(self.cdir, coin, iv)
        ts, cl, vo = [], [], []
        if p.exists():
            for line in p.open():
                if line.strip():
                    r = json.loads(line)
                    ts.append(int(r["t"]))
                    cl.append(float(r["c"]))
                    vo.append(float(r["v"]))
        self.ts[key], self.close[key], self.vol[key] = ts, cl, vo
        return bool(ts)

    def bar_index(self, key: tuple, t_ms: int) -> int | None:
        """Index of the candle with bar_t <= t_ms < bar_t + bar, or None."""
        ts = self.ts[key]
        i = bisect.bisect_right(ts, t_ms) - 1
        if i < 0 or t_ms >= ts[i] + BAR_MS[key[1]]:
            return None
        return i

    def volume_between(self, key, lo, hi):
        ts = self.ts[key]
        a, b = bisect.bisect_left(ts, lo), bisect.bisect_left(ts, hi)
        return sum(self.vol[key][a:b]), b - a


def impact_row(c, m, iv, min_bars):
    """Impact and size of one (pseudo-)metaorder, or a non-"ok" state saying why it is not measurable.

    I_tmp = eps * ln(close of the candle holding t_end / close of the candle before t_start), eps = +1 buy, -1 sell.
    I_perm is the same at t_end + 30 min. x = Q / V in base asset, V = candle volume over the 24 h before t_start.
    """
    key = (m["coin"], iv)
    if not c.load(*key):
        return {"state": "no_candles"}
    bar = BAR_MS[iv]
    if (m["t_end_ms"] - m["t_start_ms"]) < min_bars * bar:
        return {"state": "too_short"}
    i0 = c.bar_index(key, m["t_start_ms"])
    i_end = c.bar_index(key, m["t_end_ms"])
    if i0 is None or i_end is None or i0 == 0:
        return {"state": "no_baseline"}
    # baseline = close of the bar *before* the start. The start bar's own close already has
    # some of the metaorder in it.
    p0 = c.close[key][i0 - 1]
    V, nbars = c.volume_between(key, m["t_start_ms"] - VOL_MS, m["t_start_ms"])
    if V <= 0 or nbars < 0.5 * VOL_MS / bar:     # Less than half a day of bars before the start
        return {"state": "no_volume"}
    eps = 1.0 if m["side"] == "B" else -1.0
    i_tmp = eps * math.log(c.close[key][i_end] / p0)
    i_perm_idx = c.bar_index(key, m["t_end_ms"] + PERM_MS)
    i_perm = eps * math.log(c.close[key][i_perm_idx] / p0) if i_perm_idx is not None else None
    return {"state": "ok", "x": m["size"] / V, "I_tmp": i_tmp, "I_perm": i_perm,
            "wallet": m["wallet"], "coin": m["coin"], "interval": iv, "V": V, "Q": m["size"],
            "notional": m.get("notional_usd")}


def nls_beta(x: list[float], y: list[float], lo: float = 0.02, hi: float = 2.0) -> dict:
    """Fit I = A * x^beta: golden-section search on beta over [lo, hi], A in closed form for each beta.

    Bootstrap CI ~ [lo, hi] = the data don't constrain beta.
    """
    if len(x) < 3:
        return {"A": None, "beta": None, "r2": None}
    lx = [math.log(v) for v in x]

    def sse(b):
        xb = [math.exp(b * v) for v in lx]
        num = sum(yi * xi for yi, xi in zip(y, xb))
        den = sum(xi * xi for xi in xb)
        A = num / den if den > 0 else 0.0
        return sum((yi - A * xi) ** 2 for yi, xi in zip(y, xb)), A
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c1, d1 = b - g * (b - a), a + g * (b - a)
    fc, fd = sse(c1)[0], sse(d1)[0]
    for _ in range(40):
        # print(f"{a:.4f} {b:.4f} {fc:.6g} {fd:.6g}")
        if fc < fd:
            b, d1, fd = d1, c1, fc
            c1 = b - g * (b - a)
            fc = sse(c1)[0]
        else:
            a, c1, fc = c1, d1, fd
            d1 = a + g * (b - a)
            fd = sse(d1)[0]
    beta = (a + b) / 2
    s, A = sse(beta)
    ybar = sum(y) / len(y)
    sst = sum((yi - ybar) ** 2 for yi in y)
    return {"A": A, "beta": round(beta, 4), "r2": round(1 - s / sst, 4) if sst > 0 else None}


def load_metas_and_assign(data_dir):
    """Metaorders, run id, candle interval per metaorder, candle retention and the paired-metaorder flags."""
    data_dir = Path(data_dir)
    metas = [json.loads(line) for line in (data_dir / "metaorders.jsonl").open() if line.strip()]
    run_id = json.loads((data_dir / "collect_report.json").read_text())["run_id"]
    plan = json.loads((data_dir / "candles_plan.json").read_text())
    assign = {(a["wallet"], a["twap_id"]): a["interval"] for a in plan["assign"]}
    crossing = json.loads((data_dir / "crossing_report.json").read_text())
    flagged = {(p[k]["wallet"], p[k]["twap_id"]) for p in crossing["all_pairs"] for k in ("a", "b")}
    return metas, run_id, assign, plan["retention"], flagged


def truth_rows(metas, assign, flagged, c, layer, pairing, min_bars):
    rows, states = [], defaultdict(int)
    for m in metas:
        if m.get("sampling") != layer or m.get("is_boundary"):
            continue
        key = (m["wallet"], m["twap_id"])
        if pairing == "without" and key in flagged:
            continue
        iv = assign.get(key)
        if not iv:
            states["no_interval"] += 1
            continue
        r = impact_row(c, m, iv, min_bars)
        states[r["state"]] += 1
        if r["state"] == "ok":
            rows.append(r)
    return rows, dict(states)
