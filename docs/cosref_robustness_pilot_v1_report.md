# COSREF robustness pilot v1 report

This report describes a preregistered, local, scripted-agent pilot. It is not
evidence about human behavior, a causal estimate, or a reproduction of all
COSREF paper results. The paper's omega is not treated as mathematically equal
to this project's exposure-retention probability.

## Run integrity

- Preregistered config SHA-256:
  `68e02fab4cf6195efa40d609e5a8b935141fd9b6cbc533edffddd15406836f44`
- Preflight: 0.134 seconds for one excluded run; projected 30.102 seconds for
  225 formal runs, below the frozen 5,400-second stop limit.
- Formal work: 75 calibration runs and 150 evaluation runs; every one has a
  hash-checked `complete.json`. Six phase/condition checkpoints are complete,
  and `.pending` is empty.
- Actual total runtime, including preflight and analysis: 31.325 seconds.
- Remote LLM calls: 0.
- Simulation: 60 nodes, 8 timesteps, 12 active users per timestep. Calibration
  uses five seeds per candidate; evaluation uses ten disjoint seeds per
  condition and baseline.

Measured evaluation-seed mixing ranges were 0.0763--0.1174 for the strong
condition, 0.4440--0.5408 for moderate mixing, and 0.8846--0.9263 for the weak
condition. Thus all generated evaluation networks satisfy the preregistered
intervals; the report does not substitute target mu for measured mu.

## Calibration stability

Each entry below is mean +/- sample standard deviation and a fixed-seed 95%
percentile bootstrap interval over five calibration seeds. The three values
are high-risk exposure, successful risk repost, and risk cascade size. All
individual observations are retained, sorted, in
`calibration_stability.json`; `calibration_observations.jsonl` is the flat raw
record.

| condition | omega (intra, inter) | eligible | risk exposure | risk repost | cascade |
|---|---:|:---:|---:|---:|---:|
| strong | (0.2, 1.0) | yes | 252.6 +/- 9.2 [246.4, 261.0] | 24.6 +/- 2.6 [22.6, 26.6] | 26.6 +/- 2.6 [24.6, 28.6] |
| strong | (0.4, 0.8) | yes | 242.6 +/- 19.3 [229.0, 257.8] | 24.8 +/- 2.2 [23.0, 26.4] | 26.8 +/- 2.2 [25.0, 28.4] |
| strong | (0.6, 0.6) | yes | 231.2 +/- 17.6 [217.8, 244.6] | 24.0 +/- 2.5 [21.8, 25.8] | 26.0 +/- 2.5 [23.8, 27.8] |
| strong | (0.8, 0.4) | no | 241.2 +/- 24.0 [220.2, 259.8] | 25.6 +/- 2.3 [24.0, 27.4] | 27.6 +/- 2.3 [26.0, 29.4] |
| strong | (1.0, 0.2) | no | 253.2 +/- 6.8 [247.8, 258.2] | 24.8 +/- 3.3 [22.4, 27.4] | 26.8 +/- 3.3 [24.4, 29.4] |
| moderate | (0.2, 1.0) | yes | 234.8 +/- 13.3 [224.2, 244.4] | 24.2 +/- 3.9 [21.0, 27.2] | 26.2 +/- 3.9 [23.0, 29.2] |
| moderate | (0.4, 0.8) | yes | 240.8 +/- 16.7 [228.8, 255.6] | 25.0 +/- 4.1 [21.6, 27.8] | 27.0 +/- 4.1 [23.6, 29.8] |
| moderate | (0.6, 0.6) | yes | 237.2 +/- 21.8 [217.6, 252.0] | 25.0 +/- 2.3 [23.2, 26.8] | 27.0 +/- 2.3 [25.2, 28.8] |
| moderate | (0.8, 0.4) | yes | 245.2 +/- 18.3 [228.6, 257.8] | 26.2 +/- 2.5 [24.4, 28.2] | 28.2 +/- 2.5 [26.4, 30.2] |
| moderate | (1.0, 0.2) | yes | 243.2 +/- 7.0 [237.6, 248.8] | 25.4 +/- 2.3 [23.8, 27.4] | 27.4 +/- 2.3 [25.8, 29.4] |
| weak | (0.2, 1.0) | no | 229.2 +/- 4.3 [225.8, 232.6] | 24.2 +/- 3.2 [21.6, 26.6] | 26.2 +/- 3.2 [23.6, 28.6] |
| weak | (0.4, 0.8) | no | 248.2 +/- 15.5 [236.8, 261.2] | 25.4 +/- 2.6 [23.2, 27.4] | 27.4 +/- 2.6 [25.2, 29.4] |
| weak | (0.6, 0.6) | yes | 232.8 +/- 19.3 [216.4, 246.2] | 25.0 +/- 1.9 [23.4, 26.4] | 27.0 +/- 1.9 [25.4, 28.4] |
| weak | (0.8, 0.4) | yes | 231.8 +/- 24.3 [213.6, 251.6] | 24.8 +/- 3.0 [22.4, 27.2] | 26.8 +/- 3.0 [24.4, 29.2] |
| weak | (1.0, 0.2) | yes | 242.4 +/- 5.6 [238.2, 246.6] | 26.6 +/- 2.7 [24.2, 28.4] | 28.6 +/- 2.7 [26.2, 30.4] |

The full-sample selections were strong `(0.6, 0.6)`, moderate `(0.2, 1.0)`,
and weak `(0.8, 0.4)`. Their exact bootstrap selection frequencies were 0.900,
0.530, and 0.590. Leave-one-out counts were respectively 5/5 for `(0.6,0.6)`,
4/5 for `(0.2,1.0)` plus 1/5 for `(0.6,0.6)`, and 1/5 for `(0.8,0.4)` plus
4/5 for `(0.6,0.6)`.

The original directional prior is therefore not stable in this calibration.
The strong-community winner is symmetric rather than intra-prioritized. The
moderate winner is asymmetric despite the transition-band treatment, with only
0.581 bootstrap frequency for the intra direction. The weak winner has the
expected inter emphasis, but that direction appears in only 0.591 of bootstrap
selections and loses four of five leave-one-out selections to the symmetric
candidate. Exact omega stability is supported only for the symmetric strong
selection, not for the two asymmetric selections. The raw result field
`direction_stable` measures repeat selection of the winning direction; it must
not be read as alignment with the paper-guided direction.

## Evaluation

The following are absolute means over ten evaluation seeds. Benign loss is a
fraction; realized cost is the sum of candidate-level `1 - keep_probability`.

| condition | baseline | risk exposure | risk repost | cascade | benign loss | L1 cost | realized cost |
|---|---|---:|---:|---:|---:|---:|---:|
| strong | no intervention | 378.8 | 24.0 | 26.0 | 0.000 | 0.0 | 0.0 |
| strong | global throttle | 206.6 | 21.9 | 23.9 | 0.395 | 0.8 | 182.240 |
| strong | static COSREF | 231.7 | 22.2 | 24.2 | 0.000 | 0.8 | 130.464 |
| strong | dynamic COSREF | 260.2 | 22.6 | 24.6 | 0.000 | 0.8 | 113.730 |
| strong | theory-informed | 231.7 | 22.2 | 24.2 | 0.000 | 0.8 | 130.464 |
| moderate | no intervention | 384.4 | 24.0 | 26.0 | 0.000 | 0.0 | 0.0 |
| moderate | global throttle | 212.9 | 21.9 | 23.9 | 0.407 | 0.8 | 185.480 |
| moderate | static COSREF | 234.9 | 22.3 | 24.3 | 0.000 | 0.8 | 132.372 |
| moderate | dynamic COSREF | 262.2 | 23.1 | 25.1 | 0.000 | 0.8 | 115.311 |
| moderate | theory-informed | 242.7 | 24.3 | 26.3 | 0.000 | 0.8 | 140.472 |
| weak | no intervention | 387.7 | 24.0 | 26.0 | 0.000 | 0.0 | 0.0 |
| weak | global throttle | 206.4 | 22.0 | 24.0 | 0.393 | 0.8 | 184.400 |
| weak | static COSREF | 234.7 | 22.5 | 24.5 | 0.000 | 0.8 | 132.120 |
| weak | dynamic COSREF | 268.3 | 23.6 | 25.6 | 0.000 | 0.8 | 117.178 |
| weak | theory-informed | 231.3 | 23.5 | 25.5 | 0.000 | 0.8 | 138.582 |

Against no intervention, theory-informed high-risk exposure decreases were
stable in all three conditions: strong -147.1 (95% bootstrap interval -159.8
to -135.5), moderate -141.7 (-150.2 to -132.3), and weak -156.4 (-166.8 to
-147.4). Propagation did not show the same general pattern. Strong-community
risk repost and cascade differences were both -1.8 (-2.6 to -0.9), while
moderate was +0.3 (-1.1 to +1.9 for repost; -1.1 to +1.7 for cascade) and weak
was -0.5 (-1.4 to +0.4 for repost; -1.4 to +0.3 for cascade). Intervals crossing
zero are reported as uncertain.

There is no stable advantage over symmetric static COSREF. Strong results are
identical because calibration selected `(0.6,0.6)`. Moderate theory-minus-static
exposure is +7.8 (-1.4 to +19.4), but repost and cascade are each +2.0 (+0.6 to
+3.5). Weak exposure is -3.4 (-9.9 to +3.0), while repost and cascade are each
+1.0 (+0.4 to +1.5). Thus asymmetric channel allocation changed intra/inter
exposure composition without reliably improving total exposure or propagation.

For the moderate condition, theory-informed mean risk exposure (242.7) and
realized cost (140.472) are both above the corresponding strong and weak
theory-informed means. This is descriptive evidence of a harder/more costly
middle regime in this finite pilot, not a cross-condition causal estimate.

All controlled policies end at the same parameter L1 budget of 0.8, but their
realized costs are not equal. Across conditions, global throttle is about
182--185, theory-informed about 130--140, static about 130--132, and dynamic
about 114--117. Global throttle's lower risk exposure coincides with benign
exposure loss of 0.393--0.407; targeted policies show zero benign loss because
they have oracle synthetic risk labels. This comparison is consequently
favorable to risk-targeted policies and does not test classification errors.

## Interpretation and decision

Risk exposure suppression relative to no intervention is stable. Repost and
cascade improvements are stable only in the strong condition and remain
uncertain in moderate and weak conditions. More importantly, the calibrated
direction does not consistently retain the paper-guided intra/balanced/inter
priority, and precise omega selection is unstable in two of three conditions.
With only five calibration and ten evaluation seeds per cell, bootstrap
intervals describe this pilot's seed variability and are not broad external
validity guarantees.

Decision: **校准选择不稳定，暂不应接入LLM**.

The next experiment should be separately preregistered around propagation
response (for example, exposure accumulation or neighbor reinforcement) rather
than changing this pilot after seeing its outcomes.
