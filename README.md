# Glassbook

Hyperliquid's native TWAP orders tag each child fill with the parent's `twapId`, and the public info API returns those tags for any address you ask about. That is the whole trick behind Glassbook. It's the code that collects those labels, a benchmark that scores the usual metaorder reconstruction rules against them, and results on 2,529 metaorders from 77 accounts, every fill labeled with its parent.

## Why

Anyone who has worked on market impact without broker data has rebuilt metaorders from account-coded trades with some rule or other: same account, same instrument, same side, gaps below some threshold. The idea goes back at least to Chan & Lakonishok (1995) and has been in use ever since. What we couldn't find anywhere is a measurement of how often such a rule gets the parent wrong against labels from a real exchange - the checks that exist use synthetic data, or look at impact statistics and infer the error backwards from those.

Hyperliquid's TWAPs make the direct check possible. Barone & Lillo (2026) use the same labels to compare TWAPs with reconstructed metaorders; Glassbook instead hides the labels, lets a rule loose on the account's whole public tape, and then scores the result fill by fill.

The collection behind the results has the labeled metaorders with their 429,123 TWAP child fills, the same accounts' 1,458,608 other fills, and the candles. The code covers everything from collection to the impact fits. `examples/` has a 3-account sample and a template for scoring your own rule.

The short version of what came out: pair-F1 flatters the gap rules. Counted per metaorder they get a bit over half right at 5 minutes, and the misses are mostly small orders glued to other trades. Part of that gluing can't be fixed by tuning τ at all, because some accounts run several same-side TWAPs at once. And neither the paired-metaorder question nor the impact exponent gets settled with a sample this size.

## Repo layout

```
glassbook/            library code: checkpoints, paired-metaorder tests, candle layout, impact fit
glassbook/hl/         everything that talks to Hyperliquid
glassbook/metaorder/  dataset loader and the reconstruction rules / scoring
scripts/              the pipeline, roughly in the order you'd run it:
                      check_feasibility -> audit_feasibility -> collect_metaorders -> collect_mixed
                      -> collect_candles -> calibrate_pair_tests -> detect_crossing
                      -> benchmark_reconstruction -> estimate_impact -> impact_power -> summarize_findings
examples/             3-account sample and evaluate_your_heuristic.py
tests/                unit tests for the metrics
data/                 not in git; the collection scripts write here
out/                  benchmark output, also not in git
```

## Quick start

Python 3.10 or newer, standard library only (deliberately - the numerics are small enough that numpy would mostly buy a dependency). The one exception is the feasibility check, which listens to the live trade stream and needs `websockets` (`pip install -r requirements.txt`).

```bash
python3 examples/evaluate_your_heuristic.py --data examples/sample            # sample, TWAP fills only
python3 examples/evaluate_your_heuristic.py --data examples/sample --mixed    # sample, full tape
python3 -m unittest discover -s tests
```

The sample is only 3 accounts, so don't read anything into its numbers; it's there to check that the code runs.

A full collection goes into `data/`. The scripts only read public endpoints and pace themselves to the API's rate limit, and the last three take `--resume`:

```bash
python3 scripts/check_feasibility.py       # about 5 min; the trade stream it saves in data/raw/ is the wallet pool
python3 scripts/collect_metaorders.py --wallets 60 --include-top 20
python3 scripts/collect_mixed.py
python3 scripts/collect_candles.py
```

Then the analysis. The order matters: `detect_crossing.py` reads the calibration, and the benchmark and the impact fit read the paired-metaorder flags.

```bash
python3 scripts/calibrate_pair_tests.py
python3 scripts/detect_crossing.py
python3 scripts/benchmark_reconstruction.py
python3 scripts/benchmark_reconstruction.py --layers targeted --no-bl
python3 scripts/benchmark_reconstruction.py --breakdown
python3 scripts/estimate_impact.py
python3 scripts/impact_power.py
```

`benchmark_reconstruction.py` writes to `out/` by default. The other four write into `data/` unless you give them `--out`, and append to `trials.jsonl` there.

To try your own rule, edit `my_heuristic` in `examples/evaluate_your_heuristic.py`. It gets a list of `Fill(fid, wallet, coin, side, ts_ms, px, sz, crossed)` sorted by `(ts_ms, fid)` and has to return a dict mapping every `fid` to a cluster id (`fid` is unique across accounts; the scorer raises if any fid is missing). Don't peek at `twapId`, that's the answer key.

## Data

Collected from the public info API (`userTwapSliceFillsByTime`, `userFillsByTime`, `candleSnapshot`) and the public `trades` WebSocket stream. 77 accounts: 60 drawn uniformly at random from 350 TWAP users seen on the stream (the random layer), plus 17 heavy TWAP users (the targeted layer). Use only the random layer for anything that is supposed to represent the population, and don't pool the two layers.

| File | Rows in our collection | What it is |
|---|---|---|
| `data/metaorders.jsonl` | 2,529 | one labeled metaorder per line |
| `data/slices.jsonl` | 429,123 | TWAP child fills |
| `data/mixed_fills.jsonl` | 1,458,608 | the same accounts' non-TWAP fills |
| `data/candles/*.jsonl` | | candles per instrument and interval |
| `data/*.json`, `data/trials.jsonl` | | collection and analysis reports, and a log of every estimate and test |

`metaorders.jsonl`:

| Field | |
|---|---|
| `wallet`, `twap_id` | `twap_id` is unique only within an account, so key on both |
| `coin`, `side` | `side` is `B` or `A` |
| `fills` | fills, not child orders (one child can sweep several levels) |
| `size`, `notional_usd` | size in base units |
| `t_start_ms`, `t_end_ms`, `duration_s` | |
| `truncated_left` | starts within 600 s of the earliest fill the server still had for that account (`left_edge_ms`) |
| `truncated_right` | ends within 600 s of collection time |
| `is_boundary` | either of the two, left out of the results |
| `sampling`, `run_id` | `random` or `targeted`, and the collection run |

| File | Fields |
|---|---|
| `slices.jsonl` | `wallet`, `twap_id`, `tid`, `time`, `coin`, `side`, `px`, `sz`, `sampling`, `run_id` |
| `mixed_fills.jsonl` | `wallet`, `tid`, `time`, `coin`, `side`, `px`, `sz`, `oid`, `dir`, `crossed`, `zero_hash`, `is_twap`, `twap_id`, `mixed_run` |

Price and size are strings, as in the API. Both sides of a trade carry the same `tid`, so the fill key is `(wallet, tid)`. The API has no aggressor flag for TWAP fills, so the code sets `crossed = true` (TWAP children are takers).

`userFillsByTime` does not return TWAP child fills, so this file and `slices` don't overlap and together make up the account's full tape. `zero_hash` fills are almost all resting orders that a TWAP child hit, plus a few Spot Dust Conversion events. Each account has a window (`data/mixed_report.json`, `wallets[].window_start_ms`); don't use mixed fills outside it. `glassbook/metaorder/dataset.py` applies the windows, the account filter and the fill key.

Candle files are named `<instrument>_<interval>.jsonl`, with characters other than letters and digits turned into `_` and a checksum of the name appended. Each line has `coin, interval, t, o, h, l, c, v, n, candles_run`. `data/candles_plan.json` says which interval each metaorder uses: the finest one that covers 24 h before the start to 2 h after the end. The server keeps only about 5,000 candles per interval.

Main report files:

| File | |
|---|---|
| `reconstruction_twap_only.json`, `reconstruction_mixed.json` | reconstruction error on the TWAP-only and full tapes, random layer |
| `reconstruction_targeted_*.json` | the same for the targeted layer |
| `reconstruction_breakdown_*.json` | by parallel/isolated, size quartile, instrument |
| `crossing_report.json`, `pair_test_calibration.json` | paired metaorders (`verdict = calibration_failed` means undetermined) |
| `impact_report.json`, `impact_power.json` | impact exponent |
| `collect_report.json`, `mixed_report.json` | collection checks and per-account windows |

In `examples/sample/` account addresses are replaced by `w_` plus the first 16 hex characters of `sha256(salt + lowercase address)`; the salt is not published. To be clear, this is pseudonymization, not anonymization. `tid`, `twap_id`, time, instrument, price and size can all be matched against public Hyperliquid trade data, which gives the address straight back. We kept them anyway because without them nobody could check the data against the chain. Please don't use that to re-identify accounts.

## Results: reconstruction error

Rules, each in its simplest form and using only public fields:

- h1, gap runs: same account, instrument and side; consecutive fills at most τ apart.
- h2, fixed windows: same account, instrument and side; same fixed time bucket.
- h3, direction runs: same account and instrument; new metaorder on a side flip or a gap over τ.

τ runs over 60, 300, 900, 1800 and 3600 s. Metrics (`glassbook/metaorder/heuristics.py`):

- pair-F1 counts only pairs of fills that touch a scored metaorder. Pairs of two non-TWAP fills aren't scored because their parent is unknown.
- exact: the metaorder is exactly one cluster. merged: its cluster also holds another labeled metaorder. contaminated: its cluster holds non-TWAP fills. split: it ends up in several clusters.
- notional ratio: reconstructed size over true size, where each cluster's size is weighted by the share of the metaorder's notional in it. One small metaorder swallowed by a 5,000-fill cluster moves the mean of this ratio by several units, so the median, the 90th percentile and the shares above 2 and 10 are more useful.

Sample: random layer, the 57 accounts whose two sources pass the overlap check, each account's window, no truncated metaorders. That leaves 1,875 metaorders with 236,875 TWAP fills. The full tape adds 161,634 non-TWAP fills. Rules run on the whole tape and only the scoring is restricted.

Full tape:

| Rule | τ (s) | pair-F1 | exact | merged | contaminated | split | ratio median / p90 | > 2x | > 10x |
|---|---|---|---|---|---|---|---|---|---|
| h1 | 60 | 0.908 | 60.2% | 20.5% | 13.2% | 16.4% | 1.00 / 4.0 | 15.5% | 6.2% |
| h1 | 300 | 0.931 | 58.1% | 27.3% | 21.2% | 7.1% | 1.00 / 10.3 | 24.3% | 10.2% |
| h1 | 1800 | 0.910 | 47.4% | 40.1% | 28.3% | 2.4% | 1.00 / 23.5 | 38.4% | 16.4% |
| h1 | 3600 | 0.899 | 42.6% | 43.7% | 31.8% | 1.2% | 1.50 / 31.6 | 43.5% | 18.9% |
| h3 | 300 | 0.929 | 58.0% | 26.2% | 20.5% | 10.1% | 1.00 / 7.2 | 22.5% | 9.0% |
| h3 | 1800 | 0.909 | 49.8% | 35.9% | 25.7% | 6.7% | 1.00 / 13.3 | 32.8% | 12.0% |
| h2 | 3600 (best) | 0.472 | 28.4% | 37.6% | 25.7% | 39.7% | 1.00 / 13.4 | 27.5% | 11.6% |

Pair-F1 is high because big metaorders dominate it (a 5,000-fill metaorder contributes about 12.5 million pairs, a 10-fill one 45). Counting metaorders instead, the 5-minute gap rule gets 58% exactly right and the 30-minute rule 47%. The median size ratio stays at 1.00, but at 5 minutes a quarter of metaorders come out more than twice their true size, and at 30 minutes 38% do. Summed over all metaorders the overstatement is only 16-23%, because the errors sit in small metaorders. Fixed windows just don't work on this flow.

With the non-TWAP fills removed, h1 at 300 s gets pair-F1 0.962, exact 68.2%, merged 27.1%; at 1800 s, 0.953, 58.9%, 40.0%. Merging barely changes. The non-TWAP fills add contamination instead, which costs 10-12 points of exact recovery and roughly doubles the 90th-percentile size ratio.

Where the merging comes from (full tape, h1). Some accounts run several same-direction TWAPs at once on the same instrument. Their fills interleave, and no gap rule can tell them apart.

| | share of metaorders | merged at τ = 60 / 300 / 1800 / 3600 |
|---|---|---|
| overlapping a same-direction TWAP | 14.4% | 97% / 99% / 100% / 100% |
| isolated | 85.6% | 7.6% / 15.3% / 30.0% / 34.3% |

So about 14 points of the merge rate don't depend on τ at all. Among heavy users, 49% of metaorders overlap like this. For an impact study these overlapping TWAPs arguably are one parent. If they are merged before scoring, h1 at 300 s gets 66.4% exact and 15.7% merged.

Small metaorders are the hard ones. At 300 s the bottom half by notional (under about $2.5k) gets 43-49% exact and 32-36% above 2x. The top quartile (over about $20k) gets 72% exact and 13% above 2x. ETH is the easiest instrument (13% merged), BTC and HYPE sit around 33%, and PONS is an outlier at 56%. In the targeted layer (15 accounts, 411 metaorders) h1 at 300 s gets pair-F1 0.800, 29.9% exact and 61.3% merged.

We also implemented the rule specified by Barone & Lillo (2026) as closely as the data allow and applied it to the labeled TWAP metaorders: aggressive fills only, same address and instrument, same-sign runs with gaps under 30 minutes, clusters kept if they have at least 10 fills and last at most 24 h. Three differences from their setup: they count child orders and we count fills, the aggressor flag on TWAP fills comes from the documentation and the data rather than a field, and the tape has only the sampled accounts. The kept clusters cover 85.7% of the labeled metaorders and 99.8% of their notional. On those, 51.7% are exact, 36.4% merged, 24.3% contaminated, and 32.6% come out more than twice their size.

## Results: paired metaorders

Barone & Lillo (2026) estimate an impact exponent of 0.303 ± 0.002 for TWAPs, well below the square-root value of 0.5. In their joint fit of participation and duration, most of the effect runs through duration, measured in volume time. Paired metaorders could matter as well. In the raw data two accounts had metaorders of the same size, opposite sides, starting 11 s apart, on two instruments. Pairs like that have almost no net impact and would pull any estimated exponent down.

A pair is two metaorders on the same instrument, opposite sides, size within 5% (log ratio at most 0.05), starts at most 60 s apart. A second test asks whether the same two accounts pair up on two or more instruments. Each layer is tested on its own, and the "cross" group has one account from each.

Shuffling sides within each instrument doesn't work as a null here. It expects 101 pairs in the targeted layer against 7 observed, because accounts often split one order into several same-side TWAPs a few seconds apart and the shuffle turns those into fake opposite pairs. The null used here groups each account's metaorders that start within 60 s of each other and flips whole groups at random.

We found 7 pairs between one pair of heavy accounts, on 7 instruments, sizes equal to three decimals, starts 11-27 s apart. The largest random-layer account forms 17 more pairs with the same two accounts, the biggest being $14.16M against $14.16M. Paired metaorders are 16.3% of targeted-layer notional within the layer and 37.6% counting the cross pairs. In the random layer the figures are 0.0% and 6.2%, but with 17% sampling that says little about how common pairs are.

Against the grouped null the p-values are 0.010 (targeted) and 0.005 (cross), and the multi-instrument test gives 0.005. The false-positive rate on simulated data without pairs is 0.01-0.04. Five of six coincidence tests (shuffling start times, sizes, or both, in each of two groups) reject chance, but shuffling both in the targeted layer gives p = 0.069. So the result is undetermined.

TWAP children are always takers, so two TWAP children can't trade with each other. Of the roughly 23,000 trades in our sample against TWAP children, the TWAP side was the aggressor in every one. Direct execution would look like one account's TWAP hitting the other account's resting order. Between the accounts in the cluster that happens only 1-3 times, with negligible notional, so the paired metaorders executed against the public book.

They could be one entity moving positions between accounts, a market-making or basis hedge, or delta-neutral trading to build volume, for example for fee tiers or incentive programs. The data can't tell these apart.

## Results: the impact exponent

With impact measured from public candle closes and V the instrument's volume over the previous 24 h, the exponent in I = A (Q/V)^β can't be identified from the labeled metaorders. 709 random-layer metaorders were measurable, with a median Q/V of 6e-5. Mean impact was +21 bp with a standard deviation of 260 bp, and the bootstrap 95% interval for β was [0.03, 2.00], the whole search range. The 128 metaorders above $50k show a mean impact of +46 bp, but β is just as unidentified there.

A simulation with optimistic assumptions gives the sample size needed:

| price noise | n for 95% interval width ≤ 0.4 | n for width ≤ 0.2 |
|---|---|---|
| 260 bp (candle closes) | about 5,000 | 10,000-20,000 |
| 65 bp (fill-level prices) | under 700 | about 2,000 |

At n = 700 the simulated interval is 1.7-2.5 wide, which matches what we saw. Barone & Lillo work with about 465,000 TWAPs and fill-level prices from an archive, well past this threshold.

## Caveats

Things to keep in mind before leaning on the numbers:

- "Full history" means whatever the server still had when we asked. For one account the TWAP fill count dropped from 6,191 to 3,711 within 17 hours. So the collection can't be repeated exactly, a new run gets a different snapshot.
- The parent of a non-TWAP fill is unknown. There is no way around that with public data.
- Candles older than about 3 days exist only at 5-minute resolution or coarser.
- A pair of accounts is visible only if both were sampled, about 2.9% of pairs at a 17% sampling rate. Treat the random-layer pair counts as lower bounds.
- TWAPs are scheduled by the exchange. The error rates describe rules applied to TWAP parents; carry them over to hand-split or hidden orders with care.

## References

Barone, D. and Lillo, F. (2026). Trading in the Sunshine or in the Shade: Market Impact and Adverse Selection on Hyperliquid. arXiv:2606.15715.

Chan, L. K. C. and Lakonishok, J. (1995). The behavior of stock prices around institutional trades. Journal of Finance 50(4), 1147-1174.

## License

Code is MIT, see [LICENSE](LICENSE). The sample in `examples/sample/` is CC BY 4.0: use it however you like, commercially too, as long as you credit Glassbook. Its fills come from Hyperliquid's public API, and Hyperliquid's own terms for that data still apply.
