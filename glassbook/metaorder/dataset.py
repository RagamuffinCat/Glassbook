"""Dataset loader shared by scripts/benchmark_reconstruction.py and examples/, so both score the same sample.

Accounts are the wallets that pass the source overlap check and have a window in mixed_report.json; fills and
metaorders before the window start are dropped, everything else goes on the tape and focus_for() picks what's scored.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Hashable, Iterator

from .heuristics import Fill, sort_fills

MISSING = "{p} not found. The data isn't in the git repository, see README.md."


def open_jsonl(path: Path) -> Iterator[str]:
    """Yield lines from path, or from path + ".gz" if only that exists."""
    if path.exists():
        with path.open(encoding="utf-8") as f:
            yield from f
        return
    pz = path.with_name(path.name + ".gz")
    if pz.exists():
        with gzip.open(pz, "rt", encoding="utf-8") as f:
            yield from f
        return
    raise FileNotFoundError(MISSING.format(p=path))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(MISSING.format(p=path))
    return json.loads(path.read_text(encoding="utf-8"))


class Dataset:
    def __init__(self):
        self.fills: list[Fill] = []
        self.truth: dict[int, Hashable] = {}
        self.truth_oid: dict[int, Hashable] = {}      # Variant with non-TWAP fills keyed by (wallet, oid)
        self.metas: list[dict] = []                   # In-window labeled metaorders of the selected accounts
        self.meta_key_of_fid: dict[int, tuple | None] = {}
        self.wallets: set[str] = set()
        self.window_start: dict[str, int] = {}
        self.widx: dict[str, int] = {}
        self.dropped: dict[str, int] = defaultdict(int)
        self.run_id: str | None = None
        self.mixed: bool = False

    def fid(self, wallet: str, tid) -> int:
        # tid is per trade (both counterparties share it) -> fold in the wallet index.
        # tids here < 2^50
        return self.widx.setdefault(wallet, len(self.widx)) * (1 << 52) + int(tid)


def load_dataset(data_dir: Path, mixed: bool, layers: tuple[str, ...] = ("random",),
                 require_mixed_report: bool = True) -> Dataset:
    """With mixed=False, accounts and windows still come from mixed_report.json. Only non-TWAP fills are left out."""
    data_dir = Path(data_dir)
    ds = Dataset()
    ds.mixed = mixed
    metas_all = [json.loads(l) for l in open_jsonl(data_dir / "metaorders.jsonl") if l.strip()]
    ds.run_id = metas_all[0].get("run_id") if metas_all else None

    layer_wallets = {m["wallet"] for m in metas_all if m.get("sampling") in layers}
    mrep_path = data_dir / "mixed_report.json"
    if mrep_path.exists():
        mrep = read_json(mrep_path)
        ok = {d["wallet"]: int(d["window_start_ms"]) for d in mrep["wallets"]
              if d.get("state") == "ok" and d.get("source_overlap", {}).get("state") == "pass" and d.get("window_start_ms")}
        ds.wallets = layer_wallets & set(ok)
        ds.window_start = {w: ok[w] for w in ds.wallets}
    elif mixed or require_mixed_report:
        raise FileNotFoundError(MISSING.format(p=mrep_path))
    else:
        ds.wallets = set(layer_wallets)
        ds.window_start = {w: 0 for w in ds.wallets}

    for m in metas_all:
        w = m["wallet"]
        if w in ds.wallets and int(m["t_start_ms"]) >= ds.window_start[w]:
            ds.metas.append(m)
    keep_meta = {(m["wallet"], m["twap_id"]) for m in ds.metas}

    for line in open_jsonl(data_dir / "slices.jsonl"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            w = r["wallet"]
            if w not in ds.wallets or (w, r["twap_id"]) not in keep_meta:
                continue
            f = Fill(ds.fid(w, r["tid"]), w, r["coin"], r["side"], int(r["time"]), float(r["px"]), float(r["sz"]),
                     crossed=True)   # TWAP slices are always takers
        except (KeyError, TypeError, ValueError):
            ds.dropped["slices_malformed"] += 1
            continue
        ds.fills.append(f)
        ds.truth[f.fid] = ("truth", w, int(r["twap_id"]))
        ds.truth_oid[f.fid] = ds.truth[f.fid]
        ds.meta_key_of_fid[f.fid] = (w, int(r["twap_id"]))

    if mixed:
        twap_fids = set(ds.truth)
        for line in open_jsonl(data_dir / "mixed_fills.jsonl"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                w = r["wallet"]
                if w not in ds.wallets or int(r["time"]) < ds.window_start[w]:
                    continue
                f = Fill(ds.fid(w, r["tid"]), w, r["coin"], r["side"], int(r["time"]), float(r["px"]), float(r["sz"]),
                         crossed=bool(r["crossed"]) if r.get("crossed") is not None else None)
            except (KeyError, TypeError, ValueError):
                ds.dropped["mixed_malformed"] += 1
                continue
            if f.fid in twap_fids:
                ds.dropped["mixed_overlaps_twap"] += 1     # endpoints are disjoint, should stay 0
                continue
            ds.fills.append(f)
            ds.truth[f.fid] = ("noise", w, f.fid)
            oid = r.get("oid")
            ds.truth_oid[f.fid] = ("noise_oid", w, int(oid)) if oid is not None else ("noise", w, f.fid)
            ds.meta_key_of_fid[f.fid] = None
    ds.fills = sort_fills(ds.fills)
    return ds


BOUNDARY_RULES = ("all", "drop_truncated", "drop_right_truncated")


def focus_for(ds: Dataset, boundary: str = "drop_truncated", pairing: str = "with", flagged: set[tuple] | None = None):
    """Pick the scored metaorders. Returns (focus, scored keys). Clustering is unaffected.

    boundary: "all", "drop_truncated" (drop is_boundary) or "drop_right_truncated" (drop truncated_right only).
    pairing: "with", or "without" to drop flagged paired metaorders.
    """
    if boundary not in BOUNDARY_RULES:
        raise ValueError(f"boundary must be one of {BOUNDARY_RULES}, got {boundary!r}")
    flagged = flagged or set()
    keep = set()
    for m in ds.metas:
        k = (m["wallet"], m["twap_id"])
        if boundary == "drop_truncated" and m.get("is_boundary"):
            continue
        if boundary == "drop_right_truncated" and m.get("truncated_right"):
            continue
        if pairing == "without" and k in flagged:
            continue
        keep.add(k)

    def focus(tk: Hashable) -> bool:
        return isinstance(tk, tuple) and tk[0] in ("truth", "econ") and (tk[1], tk[2]) in keep
    return focus, keep
