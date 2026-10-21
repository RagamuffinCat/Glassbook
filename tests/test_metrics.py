"""Tests for the evaluation metrics. Mostly small hand-computed cases, plus a brute-force cross-check
of the pair counting and an input-order invariance test. Stdlib unittest, nothing to install.

    python3 -m unittest discover -s tests
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glassbook.metaorder.heuristics import (Fill, _pair_counts, _pair_counts_bruteforce, economic_truth,
                                            evaluate, h1_gap_runs, quantile, sort_fills)


def mk(fid, wallet="w", coin="BTC", side="B", ts=0, px=100.0, sz=1.0):
    return Fill(fid, wallet, coin, side, ts, px, sz, crossed=True)


class PerfectHeuristic(unittest.TestCase):
    def setUp(self):
        self.fills = [mk(i, ts=1000 * i) for i in range(10)] + [mk(100 + i, ts=60_000 * 10 + 1000 * i) for i in range(5)]
        self.truth = {i: ("truth", "w", 1) for i in range(10)}
        self.truth.update({100 + i: ("truth", "w", 2) for i in range(5)})

    def test_perfect(self):
        m = evaluate(self.truth, dict(self.truth), fills=self.fills)
        self.assertEqual(m["pair_f1"], 1.0)
        self.assertEqual(m["pair_f1_focus"], 1.0)
        self.assertEqual(m["jaccard_focus"], 1.0)
        self.assertEqual(m["exact_recovered_rate"], 1.0)
        self.assertEqual(m["merge_rate"], 0.0)
        self.assertEqual(m["split_rate"], 0.0)
        self.assertEqual(m["notional_ratio_median"], 1.0)
        self.assertEqual(m["notional_ratio_aggregate"], 1.0)
        self.assertEqual(m["count_ratio_median"], 1.0)


class HandComputed(unittest.TestCase):
    def test_merge_two_metaorders_plus_noise(self):
        # Metaorder 1 is 10 fills of $100, metaorder 2 is 5 fills of $100, plus one $700 non-TWAP fill, all in one cluster
        fills = ([mk(i, ts=1000 * i) for i in range(10)] + [mk(100 + i, ts=1000 * (20 + i)) for i in range(5)]
                 + [mk(200, side="A", ts=3000, sz=7.0)])
        truth = {i: ("truth", "w", 1) for i in range(10)}
        truth.update({100 + i: ("truth", "w", 2) for i in range(5)})
        truth[200] = ("noise", "w", 200)
        pred = {f: "one" for f in truth}
        m = evaluate(truth, pred, fills=fills)
        # All pairs: TP = C(10,2) + C(5,2) = 55, predicted = C(16,2) = 120, F1 = 2*55 / (55+120)
        self.assertAlmostEqual(m["pair_f1"], 2 * 55 / (55 + 120), places=4)
        # Focus: predicted = C(16,2) - C(1,2) = 120, since only one fill is outside the scored metaorders
        self.assertAlmostEqual(m["pair_f1_focus"], 2 * 55 / (55 + 120), places=4)
        self.assertEqual(m["merge_twap_rate"], 1.0)
        self.assertEqual(m["contam_nontwap_rate"], 1.0)
        self.assertEqual(m["merge_rate"], 1.0)
        self.assertEqual(m["exact_recovered_rate"], 0.0)
        # Cluster notional 2200, so ratios are 2200/1000 = 2.2 and 2200/500 = 4.4.
        # Median 3.3, aggregate 4400/1500.
        self.assertAlmostEqual(m["notional_ratio_median"], 3.3, places=3)
        self.assertAlmostEqual(m["notional_ratio_aggregate"], 4400 / 1500, places=3)
        self.assertEqual(m["notional_ratio_share_gt2"], 1.0)
        self.assertEqual(m["notional_ratio_share_gt10"], 0.0)

    def test_split(self):
        fills = [mk(i, ts=1000 * i) for i in range(6)]
        truth = {i: ("truth", "w", 1) for i in range(6)}
        pred = {i: ("a" if i < 3 else "b") for i in range(6)}
        m = evaluate(truth, pred, fills=fills)
        self.assertEqual(m["split_rate"], 1.0)
        self.assertEqual(m["merge_rate"], 0.0)
        self.assertEqual(m["exact_recovered_rate"], 0.0)
        # True pairs C(6,2) = 15, predicted 2*C(3,2) = 6, TP = 6. P = 1, R = 0.4, F1 = 0.5714
        self.assertAlmostEqual(m["pair_f1"], 2 * 6 / (15 + 6), places=4)
        self.assertAlmostEqual(m["notional_ratio_median"], 0.5, places=3)

    def test_noise_only_pairs_not_counted_in_focus_metric(self):
        # Two non-TWAP fills grouped together count against all-pairs F1 but not the focus metric
        fills = [mk(1, ts=0), mk(2, ts=1000), mk(300, ts=5000), mk(301, ts=6000)]
        truth = {1: ("truth", "w", 1), 2: ("truth", "w", 1), 300: ("noise", "w", 300), 301: ("noise", "w", 301)}
        pred = {1: "t", 2: "t", 300: "n", 301: "n"}
        m = evaluate(truth, pred, fills=fills)
        self.assertLess(m["pair_f1"], 1.0)
        self.assertEqual(m["pair_f1_focus"], 1.0)
        self.assertEqual(m["contam_nontwap_rate"], 0.0)


class OrderInvariance(unittest.TestCase):
    def test_shuffled_input_same_result(self):
        rng = random.Random(3)
        fills = []
        truth = {}
        for w in range(5):
            for k in range(4):
                base = rng.randint(0, 10**7)
                for i in range(8):
                    fid = (w * 100 + k * 10 + i)
                    fills.append(mk(fid, wallet=f"w{w}", ts=base + i * 2000))
                    truth[fid] = ("truth", f"w{w}", k)
        a = evaluate(truth, h1_gap_runs(sort_fills(fills), tau_s=5), fills=fills)
        shuffled = list(fills)
        rng.shuffle(shuffled)
        b = evaluate(truth, h1_gap_runs(sort_fills(shuffled), tau_s=5), fills=shuffled)
        self.assertEqual(a, b)


class Coverage(unittest.TestCase):
    def test_missing_fid_raises(self):
        truth = {1: ("truth", "w", 1), 2: ("truth", "w", 1)}
        with self.assertRaises(ValueError):
            evaluate(truth, {1: "a"})

    def test_unhashable_raises(self):
        truth = {1: ("truth", "w", 1)}
        with self.assertRaises(TypeError):
            evaluate(truth, {1: ["list"]})


class Helpers(unittest.TestCase):
    def test_pair_counts_matches_bruteforce(self):
        rng = random.Random(1)
        fids = list(range(200))
        a = {f: rng.randint(0, 20) for f in fids}
        b = {f: rng.randint(0, 20) for f in fids}
        self.assertEqual(_pair_counts(a, b, fids), _pair_counts_bruteforce(a, b, fids))

    def test_quantile_linear(self):
        self.assertEqual(quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.9), 4.6)
        self.assertIsNone(quantile([], 0.5))

    def test_economic_truth_merges_overlapping_same_side(self):
        fills = [mk(1, ts=0), mk(2, ts=10_000), mk(3, ts=5_000), mk(4, ts=20_000), mk(5, side="A", ts=6_000)]
        truth = {1: ("truth", "w", 1), 2: ("truth", "w", 1), 3: ("truth", "w", 2), 4: ("truth", "w", 2), 5: ("truth", "w", 3)}
        e = economic_truth(truth, fills)
        self.assertEqual(e[1], e[3])          # Spans [0, 10s] and [5s, 20s] overlap
        self.assertNotEqual(e[1], e[5])       # Opposite side


if __name__ == "__main__":
    unittest.main()
