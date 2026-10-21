#!/usr/bin/env python3
"""Score your own metaorder-reconstruction rule against the exchange labels.

    python3 examples/evaluate_your_heuristic.py --data examples/sample            # bundled sample, TWAP-only
    python3 examples/evaluate_your_heuristic.py --data examples/sample --mixed    # bundled sample, mixed tape
    python3 examples/evaluate_your_heuristic.py --data data --mixed --tau 1800    # full collection in data/

my_heuristic gets a list of Fill(fid, wallet, coin, side, ts_ms, px, sz, crossed) sorted by (ts_ms, fid)
and returns a dict mapping every fid to a hashable cluster id. fid is unique across accounts. Loader
and metrics are the same as in scripts/benchmark_reconstruction.py.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glassbook.metaorder.dataset import focus_for, load_dataset
from glassbook.metaorder.heuristics import Fill, evaluate, h1_gap_runs, sort_fills


def my_heuristic(fills, tau_s):
    """Put your rule here. Default is H1 gap runs at tau_s, so running this unchanged just
    reproduces the h1 row of the benchmark."""
    return h1_gap_runs(fills, tau_s=tau_s)


def main():
    ap = argparse.ArgumentParser(description="score your own metaorder-reconstruction rule")
    ap.add_argument("--data", type=Path, default=ROOT / "examples" / "sample")
    ap.add_argument("--mixed", action="store_true", help="include the accounts' non-TWAP fills (real mixed tape)")
    ap.add_argument("--tau", type=float, default=300)
    ap.add_argument("--layers", default="random")
    args = ap.parse_args()
    data_dir = args.data.resolve()
    if data_dir == (ROOT / "examples" / "sample").resolve():
        print(f"data: {data_dir} (3-account sample, --data data for the full set)")
    else:
        print(f"data: {data_dir}")
    try:
        ds = load_dataset(args.data, mixed=args.mixed, layers=tuple(args.layers.split(",")))
    except FileNotFoundError as e:
        sys.exit(str(e))
    focus, keep = focus_for(ds, "drop_truncated", "with")
    fills = sort_fills(ds.fills)
    n_noise = sum(1 for k in ds.truth.values() if k[0] == "noise")
    print(f"accounts {len(ds.wallets)}, fills {len(fills):,} (non-TWAP {n_noise:,}), scored metaorders {len(keep):,}"
          f", dropped malformed rows {dict(ds.dropped) or 0}")
    pred = my_heuristic(fills, args.tau)
    m = evaluate(ds.truth, pred, fills=fills, focus=focus)
    print(f"scored fills {m['n_scored_fills']:,} of {m['n_fills']:,}")
    print(f"tau={args.tau:.0f}s, pair F1 (pairs touching a scored metaorder) {m['pair_f1_focus']:.3f}, Jaccard {m['jaccard_focus']:.3f}"
          f", pair F1 (all pairs) {m['pair_f1']:.3f}")
    print(f"  exact {m['exact_recovered_rate']:.1%}, merged with another TWAP {m['merge_twap_rate']:.1%}"
          f", contaminated by non-TWAP {m['contam_nontwap_rate']:.1%}, split {m['split_rate']:.1%}")
    print(f"  notional ratio: median {m['notional_ratio_median']:.2f}, p90 {m['notional_ratio_p90']:.2f}"
          f", geomean {m['notional_ratio_geomean']:.2f}, >2x {m['notional_ratio_share_gt2']:.1%}"
          f", >10x {m['notional_ratio_share_gt10']:.1%}, aggregate {m['notional_ratio_aggregate']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
