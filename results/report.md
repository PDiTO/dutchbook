# dutchbook results

Bank seed 20261103: 176 propositions, 75 items (complement 27, monotonicity 14, partition 14, conjunction 6, conditional 6, paraphrase 8).

All arbitrage numbers are cents of guaranteed profit per $1 of total stake, mean over the six families, with 95% topic-cluster bootstrap intervals.

## Headline

| forecaster | asked once | noise only, asked once | averaged | noise only, averaged | excess over noise | L2 distance | noise p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-opus-5-5 | 1.0 [0.9, 1.2] | 0.8 [0.7, 0.8] | 0.8 [0.6, 1.0] | 0.3 [0.3, 0.4] | 0.4 [0.3, 0.6] | 1.4 [1.1, 1.8] | 0.001 |
| claude-sonnet-5-5 | 1.4 [1.1, 1.6] | 0.8 [0.7, 0.8] | 1.1 [0.9, 1.4] | 0.3 [0.3, 0.4] | 0.8 [0.5, 1.0] | 2.1 [1.7, 2.6] | 0.001 |
| claude-haiku-5-5 | 2.4 [1.9, 2.8] | 2.2 [1.9, 2.5] | 1.9 [1.4, 2.4] | 1.0 [0.8, 1.2] | 0.9 [0.5, 1.4] | 3.5 [2.6, 4.5] | 0.001 |
| gpt-6.1-sol | 1.3 [1.1, 1.5] | 0.7 [0.6, 0.7] | 1.1 [0.9, 1.3] | 0.3 [0.3, 0.3] | 0.8 [0.6, 1.0] | 2.3 [1.8, 2.7] | 0.001 |
| coherent | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.0 [-0.0, -0.0] | 0.0 [0.0, 0.0] | 1.000 |
| coherent + noise | 0.5 [0.4, 0.6] | 0.6 [0.5, 0.6] | 0.2 [0.2, 0.3] | 0.2 [0.2, 0.3] | -0.0 [-0.1, 0.0] | 0.4 [0.3, 0.5] | 0.867 |
| random | 22.0 [20.4, 23.5] | 15.4 [14.8, 16.0] | 12.1 [10.6, 13.6] | 6.8 [6.5, 7.1] | 5.3 [3.9, 6.7] | 24.2 [21.3, 27.0] | 0.003 |

## Calls

| model | calls | transport errors | replies | ok | refusal | no tag | multiple tags | bad number | tool events |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-opus-5-5 | 880 | 0 | 880 | 880 | 0 | 0 | 0 | 0 | 0 |
| claude-sonnet-5-5 | 880 | 0 | 880 | 880 | 0 | 0 | 0 | 0 | 0 |
| claude-haiku-5-5 | 880 | 0 | 880 | 880 | 0 | 0 | 0 | 0 | 0 |
| gpt-6.1-sol | 883 | 3 | 880 | 880 | 0 | 0 | 0 | 0 | 0 |

## Test-retest

| model | mean SD | median SD | identical every time | ICC | mean abs diff |
| :--- | ---: | ---: | ---: | ---: | ---: |
| claude-opus-5-5 | 1.6 | 1.3 | 5% | 0.99 | 1.9 |
| claude-sonnet-5-5 | 1.7 | 1.5 | 4% | 0.99 | 2.0 |
| claude-haiku-5-5 | 3.6 | 2.4 | 3% | 0.95 | 4.1 |
| gpt-6.1-sol | 1.5 | 1.3 | 3% | 1.00 | 1.8 |
| coherent | 0.0 | 0.0 | 100% | 1.00 | 0.0 |
| coherent + noise | 1.2 | 1.1 | 7% | 1.00 | 1.4 |
| random | 28.0 | 28.2 | 0% | 0.00 | 33.4 |

## By family

### asked once

| family | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 | gpt-6.1-sol |
| :--- | ---: | ---: | ---: | ---: |
| complement | 1.9 [1.5, 2.4] | 2.2 [1.6, 3.0] | 3.1 [2.1, 4.2] | 1.8 [1.5, 2.1] |
| monotonicity | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 1.3 [0.1, 3.1] | 0.0 [0.0, 0.1] |
| partition | 1.8 [1.2, 2.6] | 2.8 [2.0, 3.7] | 4.8 [3.2, 6.5] | 3.3 [2.4, 4.4] |
| conjunction | 0.8 [0.4, 1.2] | 1.3 [0.9, 1.7] | 1.4 [0.9, 1.9] | 1.0 [0.7, 1.3] |
| conditional | 0.5 [0.3, 0.7] | 0.6 [0.3, 0.9] | 0.7 [0.3, 1.2] | 0.5 [0.4, 0.6] |
| paraphrase | 1.2 [0.8, 1.6] | 1.2 [0.8, 1.6] | 2.8 [1.9, 3.6] | 1.1 [0.8, 1.5] |

### averaged

| family | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 | gpt-6.1-sol |
| :--- | ---: | ---: | ---: | ---: |
| complement | 1.6 [1.2, 2.0] | 2.1 [1.4, 2.9] | 2.5 [1.5, 3.5] | 1.5 [1.1, 1.9] |
| monotonicity | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9 [0.0, 2.4] | 0.0 [0.0, 0.0] |
| partition | 1.7 [1.1, 2.5] | 2.7 [1.8, 3.6] | 4.5 [2.9, 6.2] | 3.3 [2.3, 4.3] |
| conjunction | 0.5 [0.1, 1.1] | 0.9 [0.4, 1.4] | 1.0 [0.2, 1.8] | 0.9 [0.6, 1.3] |
| conditional | 0.2 [0.1, 0.3] | 0.4 [0.1, 0.8] | 0.3 [0.1, 0.5] | 0.4 [0.3, 0.6] |
| paraphrase | 0.5 [0.3, 0.8] | 0.9 [0.4, 1.3] | 2.1 [1.1, 3.0] | 0.6 [0.2, 0.9] |

### averaged minus noise only

| family | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 | gpt-6.1-sol |
| :--- | ---: | ---: | ---: | ---: |
| complement | 1.1 [0.7, 1.5] | 1.5 [0.8, 2.3] | 1.1 [0.2, 2.0] | 1.0 [0.6, 1.4] |
| monotonicity | -0.0 [-0.1, -0.0] | -0.0 [-0.1, -0.0] | 0.3 [-0.3, 1.4] | -0.0 [-0.0, -0.0] |
| partition | 1.4 [0.8, 2.2] | 2.4 [1.5, 3.3] | 3.7 [2.0, 5.3] | 3.0 [2.1, 4.1] |
| conjunction | 0.2 [-0.2, 0.8] | 0.5 [-0.0, 1.0] | 0.2 [-0.6, 0.9] | 0.6 [0.3, 1.0] |
| conditional | -0.1 [-0.3, 0.1] | 0.1 [-0.2, 0.4] | -0.5 [-0.6, -0.3] | 0.1 [-0.0, 0.3] |
| paraphrase | 0.0 [-0.2, 0.2] | 0.3 [-0.1, 0.8] | 0.8 [-0.2, 1.7] | 0.1 [-0.3, 0.4] |

## Items beyond noise (Benjamini-Hochberg, q = 0.05)

| family | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 | gpt-6.1-sol |
| :--- | ---: | ---: | ---: | ---: |
| complement | 10 / 27 | 14 / 27 | 2 / 27 | 13 / 27 |
| monotonicity | 0 / 14 | 0 / 14 | 1 / 14 | 0 / 14 |
| partition | 10 / 14 | 11 / 14 | 8 / 14 | 11 / 14 |
| conjunction | 1 / 6 | 3 / 6 | 0 / 6 | 3 / 6 |
| conditional | 0 / 6 | 1 / 6 | 0 / 6 | 0 / 6 |
| paraphrase | 0 / 8 | 2 / 8 | 1 / 8 | 0 / 8 |
| all | 21 / 75 | 31 / 75 | 12 / 75 | 27 / 75 |

## Which way the sums lean

Mean of the averaged answers summed over each complement pair or partition. Coherent answers sum to 1. Range in brackets; share of items summing above 1 after.

| family / topic | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 | gpt-6.1-sol |
| :--- | ---: | ---: | ---: | ---: |
| complement/dated event | 1.00 [0.94, 1.10], 33% | 0.96 [0.90, 1.00], 0% | 0.97 [0.92, 1.09], 17% | 1.04 [1.01, 1.07], 100% |
| complement/numeric | 0.97 [0.91, 1.01], 12% | 0.96 [0.86, 1.05], 12% | 0.95 [0.81, 1.01], 12% | 1.02 [0.98, 1.05], 94% |
| complement/who wins | 0.98 [0.94, 1.00], 20% | 0.98 [0.95, 1.00], 0% | 0.99 [0.92, 1.03], 40% | 0.98 [0.96, 1.00], 0% |
| partition/numeric | 1.15 [1.06, 1.28], 100% | 1.20 [1.11, 1.31], 100% | 1.38 [1.26, 1.63], 100% | 1.29 [1.23, 1.36], 100% |
| partition/who wins | 0.97 [0.90, 1.02], 50% | 0.93 [0.80, 1.03], 17% | 0.91 [0.71, 1.01], 17% | 1.07 [1.00, 1.14], 83% |

## Paired differences (cents per $1)

| comparison | difference |
| :--- | ---: |
| claude-opus-5-5 - claude-sonnet-5-5: single | -0.3 [-0.5, -0.1] |
| claude-opus-5-5 - claude-sonnet-5-5: averaged | -0.4 [-0.6, -0.1] |
| claude-opus-5-5 - claude-sonnet-5-5: excess | -0.4 [-0.6, -0.1] |
| claude-opus-5-5 - claude-haiku-5-5: single | -1.3 [-1.8, -0.9] |
| claude-opus-5-5 - claude-haiku-5-5: averaged | -1.1 [-1.6, -0.7] |
| claude-opus-5-5 - claude-haiku-5-5: excess | -0.5 [-0.9, -0.1] |
| claude-opus-5-5 - gpt-6.1-sol: single | -0.3 [-0.5, -0.1] |
| claude-opus-5-5 - gpt-6.1-sol: averaged | -0.3 [-0.5, -0.2] |
| claude-opus-5-5 - gpt-6.1-sol: excess | -0.4 [-0.6, -0.2] |
| claude-sonnet-5-5 - claude-haiku-5-5: single | -1.0 [-1.5, -0.5] |
| claude-sonnet-5-5 - claude-haiku-5-5: averaged | -0.8 [-1.3, -0.2] |
| claude-sonnet-5-5 - claude-haiku-5-5: excess | -0.1 [-0.6, 0.3] |
| claude-sonnet-5-5 - gpt-6.1-sol: single | 0.1 [-0.2, 0.3] |
| claude-sonnet-5-5 - gpt-6.1-sol: averaged | 0.0 [-0.2, 0.3] |
| claude-sonnet-5-5 - gpt-6.1-sol: excess | -0.0 [-0.3, 0.3] |
| claude-haiku-5-5 - gpt-6.1-sol: single | 1.1 [0.6, 1.5] |
| claude-haiku-5-5 - gpt-6.1-sol: averaged | 0.8 [0.4, 1.3] |
| claude-haiku-5-5 - gpt-6.1-sol: excess | 0.1 [-0.2, 0.5] |

## Polymarket (polymarket-2026-11-03.json)

| shape | markets | mid: mean | mid: median | mid: share incoherent | executable books | executable: max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| partition/all | 714 | 0.78 | 0.17 | 65% | 21 | 0.62 |
| partition/tight | 350 | 0.19 | 0.06 | 55% | 19 | 0.20 |
| ladder/all | 183 | 1.83 | 0.00 | 43% | 7 | 0.50 |
| ladder/tight | 58 | 0.19 | 0.00 | 38% | 5 | 0.50 |
