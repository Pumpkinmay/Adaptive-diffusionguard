# COSREF propagation-signal audit

This audit describes the existing scripted-agent experiment before the
robustness pilot was configured. It does not change agent behavior.

## 1. Repost behavior after risky exposure

`adaptive_diffusionguard/theory/experiment.py::_scripted_action` partitions the
visible feed by root-post risk. For each risky post, in feed order, it computes

```text
(user_id * 7 + timestep * 3 + seed) % 4 == 0
```

When true, the deterministic preference is `repost` then `report`; otherwise it
is `report` then `repost`. This is not a sampled probability, although over
uniform residues the first-choice repost share is nominally one quarter. If the
preferred action is unavailable because the user already performed it, the
other action is attempted. At most one successful action is taken per refresh.
If no actionable risky post is found, a benign repost is attempted when
`(user_id + timestep + seed) % 5 == 0`.

## 2. Missing propagation-response dependencies

The choice formula does not use cumulative exposures, number or fraction of
active neighbors, number of prior impressions, community relation, risk score
magnitude, cascade depth, or social reinforcement. Community structure affects
which candidates become visible, but not the conditional repost rule once a
post is visible.

## 3. Why lower exposure need not yield a visibly smaller cascade

Several mechanisms weaken the exposure-to-cascade signal:

- the agent takes only one action per refresh and usually prefers `report`;
- the deterministic residue rule, rather than exposure accumulation, controls
  whether repost is attempted first;
- a previously reported risky post can later fall through to repost;
- only a small number of users are active per timestep;
- suppressing one candidate can expose a different candidate in the bounded
  feed, so candidate composition changes rather than simply shrinking;
- OASIS repost constraints cap repeated reposts by the same user/root;
- three original timesteps offered little opportunity for descendants to be
  recommended and acted on across several generations.

Consequently, a strong decrease in impressions can coexist with a small or
noisy change in successful repost count and terminal cascade size.

## 4. Is three timesteps sufficient?

It can create descendants, because every timestep rebuilds the recommendation
table from current posts. It is not sufficient evidence for stable multi-hop
behavior: a post made late in timestep 2 or 3 has at most one or zero later
recommendation rounds. The pilot therefore preregisters eight timesteps while
keeping the response rule unchanged.

## 5. Cascade-size implementation

The generic `metrics.diffusion.CascadeSize` helper counts only the root and
rows whose `original_post_id` directly equals that root. The theory experiment
does not use that helper. It iterates over every post and calls
`AdaptiveDiffusionPlatform._root_post`, which follows the complete
`original_post_id` chain. Its `risk_cascade_size` therefore includes risky
roots and all repost/quote descendants, including multi-generation chains.
The count is a number of post records, not unique adopters.

## 6. Effect of reports

OASIS `report_post` increments `post.num_reports`, inserts a report record, and
adds a trace. It does not delete, downrank, or block the post. Once the report
threshold is reached, feed formatting adds a warning string. The current
scripted agent does not inspect that warning, so reports do not directly reduce
later visibility or adoption in this experiment.

## 7. Why candidate counts diverge between policies

The policies start from the same network and seed posts, but shown feeds cause
different reports/reposts. Reposts become new rows in the OASIS `post` table.
At each timestep `update_rec_table()` rebuilds recommendations from the current
post and trace state, and followed-user content is also included. Once action
histories diverge, the available post population, recommendation table, root
duplicates, and eligibility checks diverge. Thus later candidate counts are
not mechanically identical across policies.

## 8. What fixed seeds do and do not pair

For a condition and seed, the experiment pairs:

- directed SBM generation;
- community assignments and follow arcs;
- four initial synthetic posts and their risk labels;
- active-user schedule;
- candidate-generator and exposure-policy pseudorandom streams at their
  initial state.

It cannot guarantee event-by-event pairing after the first policy-dependent
exposure. Different actions change the post and trace tables, causing later RNG
calls and candidate sets to represent different events. Results remain paired
by ex-ante seed, not by every realized impression.

## 9. Parameter budget versus realized intervention cost

`project_budget=0.8` constrains the parameter displacement

```text
(1 - omega_intra) + (1 - omega_inter) <= 0.8
```

It is dimensionless and independent of traffic volume. The recorded
`realized_intervention_cost` is

```text
sum(1 - keep_probability)
```

over all candidate impressions. It is an expected-suppression mass and grows
with the number, risk, and intra/inter composition of candidates. It is not the
paper's exponential cost, is not normalized by candidate count, and is not
directly comparable across runs with different candidate volumes without also
reporting that volume. The pilot reports both quantities and paired
differences.

## 10. Known-risk labels and benign-loss asymmetry

The static and theory-informed COSREF policy uses the experiment's known risk
score in `1 - risk * (1 - omega)`. A zero-risk candidate is therefore always
kept. The uniform global-throttle baseline does not receive that exemption and
suppresses benign candidates too. This deliberately demonstrates a risk/benign
tradeoff, but it advantages risk-targeted policies through oracle labels. It is
not a test of risk-classifier errors, uncertain labels, or deployable
moderation. Benign-loss comparisons must be interpreted under this synthetic
known-truth assumption.

## Audit implication

The robustness pilot can test whether the existing deterministic mechanism
produces stable exposure and propagation differences over more networks and
timesteps. It cannot validate cumulative social reinforcement or human
behavior. If exposure effects stabilize while repost/cascade effects do not,
the appropriate next step is a separately preregistered propagation-response
design, not post-hoc tuning of this agent.
