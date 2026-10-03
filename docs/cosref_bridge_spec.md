# Minimal COSREF theory bridge

This document separates (A) statements verified in the article or its official
v1.0.0 code from (B) project adaptations used by Adaptive DiffusionGuard. The
bridge is deliberately narrower than a reproduction of the paper.

## Primary sources and pinned artifacts

- Article: Chen et al., *Community structure-regulation coupling reveals
  optimal information diffusion*, Nature Communications 17, 4879 (2026),
  [DOI 10.1038/s41467-026-73665-1](https://doi.org/10.1038/s41467-026-73665-1).
- Supplementary PDF: Springer Nature object
  `41467_2026_73665_MOESM1_ESM.pdf`, SHA-256
  `358bb1d34434d8549166c7ad01d366f32a6b87c8537c3e1f9fd915f1e2d0fee1`.
- Official software: [GitHub tag v1.0.0](https://github.com/fanjingfang/Epidemic/tree/v1.0.0),
  archived commit `9629dd7a7adecaadfffd53f2ae0f3a28a75a54eb`.
- Immutable archive: [Zenodo 10.5281/zenodo.19841214](https://doi.org/10.5281/zenodo.19841214),
  downloaded ZIP SHA-256
  `c78f746b3b4dc3566f63e6015ff9019ec8dc5e077d93bd22bfcb2a324109a253`.

Relevant official-code hashes are preserved in
`runs/cosref-reference-validation/reference-oracle/summary.json`. The official
archive was inspected and executed from `/private/tmp`; none of it is vendored
into the production package.

## A. Verified paper model

### Community mixing

For the paper's undirected two-module contact network, Eq. (1) defines

\[
\mu = \frac{z_{\mathrm{inter}}}
            {z_{\mathrm{inter}}+z_{\mathrm{intra}}}
     = \frac{z_{\mathrm{inter}}}{z}.
\]

Equivalently for that undirected construction, it is the fraction of edges
that join different modules. Small \(\mu\) means pronounced/strong modular
structure; large \(\mu\) means weak modular structure. The article identifies
\(\mu=0.5\) as random-like, \(\mu=0\) as fully separated modules, and
\(\mu=1\) as an all-inter-edge bipartite limit.

### Regulation parameters and threshold dynamics

Each node has state \(x_i\in\{0,1\}\), susceptible or adopted. Initial adopters
with global fraction \(\rho_0\) are placed in one module. The article states
that nodes are updated synchronously. A susceptible node of total degree \(k\)
adopts under Eq. (2) when

\[
\mathcal R(\mathbf m,\boldsymbol\omega,\theta,k)=
\begin{cases}
1,& \mathbf m\!\cdot\!\boldsymbol\omega>\theta k,\\
0,& \text{otherwise},
\end{cases}
\]

where \(\mathbf m=(m_{\mathrm{intra}},m_{\mathrm{inter}})\) counts adopted
neighbors and
\(\boldsymbol\omega=(\omega_{\mathrm{intra}},\omega_{\mathrm{inter}})\).
The strict `>` is present both in Eq. (2) and in the official code.
\(\omega_{\mathrm{intra}}\) and \(\omega_{\mathrm{inter}}\) are effective
within- and between-community transmissibilities/influence weights. Lower
values mean stronger regulation: zero is complete suppression and one is
unrestricted transmission. Thus \((1-\omega_{\mathrm{intra}},
1-\omega_{\mathrm{inter}})\) is regulatory intensity. Adoption is progressive
in the main susceptible-adopted model; the paper separately studies a
recurrent SIS extension in Supplementary Section I.

The main text's update statement is the normative definition. The official
C++ Monte Carlo implementation (`ER-ER-ER/.../network.cpp`) computes the same
weighted threshold but propagates newly adopted nodes through a FIFO queue.
That is an operationally asynchronous implementation. In a static,
irreversible, monotone threshold process it is intended to reach the closure of
the activated set, but this bridge does not claim step-by-step equivalence to
the synchronous trajectory.

### Diffusion regimes

The article and Fig. 2 classify the final global adoption density
\(\rho_\infty\) as:

- non-diffusion: spread remains near the seed density,
  \(\rho_\infty\approx\rho_0\);
- localized diffusion: the seeded one of two equal modules saturates,
  \(\rho_\infty\approx0.5\);
- global diffusion: system-wide adoption, \(\rho_\infty\approx1\).

These are phase descriptions, not universal numeric classifiers. The official
Fig. 3 plotting script uses `rho <= 0.3` to extract the plotted controllable
boundary; our small reference slice records that code-level cutoff explicitly.

### Intervention cost and optimum

The controllable region \(\mathcal U_{\mathrm{free}}\) contains parameter
combinations for which diffusion does not percolate. Equation (4) defines

\[
\boldsymbol\omega_o=
\operatorname*{arg\,min}_{\boldsymbol\omega\in\mathcal U_{\mathrm{free}}}
\mathcal F(\boldsymbol\omega).
\]

Equation (5), confirmed by `MOCOF/scripts/run_boundary_plot.py`, is

\[
\mathcal F(\boldsymbol\omega)=
\frac{\exp[-(\omega_{\mathrm{intra}}+\omega_{\mathrm{inter}})]-e^{-2}}
     {1-e^{-2}}.
\]

It ranges from zero at `(1, 1)` to one at `(0, 0)`. It is symmetric in the two
channels, and its iso-cost lines have slope -1. The optimum is the lowest-cost
point in the controllable region (geometrically the tangency described in the
main text and Supplementary Fig. 16). Supplementary Fig. 4 reports that linear
and quadratic alternatives retain the qualitative non-monotone cost profile;
this is not a statement that their numerical costs are interchangeable.

### Strong versus weak community structure

Main Fig. 3 and Supplementary Fig. 16 are the direct basis for the directional
prior:

- \(\mu<0.5\): intra-community reinforcement dominates, so the optimum favors
  lower \(\omega_{\mathrm{intra}}\) (stronger intra control).
- \(\mu>0.5\): cross-community bridges are more abundant, so the optimum
  favors lower \(\omega_{\mathrm{inter}}\) (stronger inter control).

Supplementary Fig. 16 reports continuous-grid optima `(0.00, 0.85)`,
`(0.23, 0.34)`, `(0.38, 0.21)`, and `(0.78, 0.10)` for
\(\mu=0.2,0.4,0.6,0.8\), respectively. Near 0.5 the present bridge uses a
declared tolerance band (default `|mu-0.5| <= 0.05`) and applies no one-sided
prior. The band is a project safeguard, not a paper formula.

The tree-like equations are given in Methods Eqs. (7)-(8); the explicit
multi-module mean-field and three-module forms are in Supplementary Sections
II-III (Eqs. S6-S9 and Supplementary Fig. 15). They are documented but not
reimplemented here.

## B. Adaptive DiffusionGuard project adaptation

### Directed follower network

The paper studies undirected contacts, whereas an OASIS follow relation is a
directed arc. `compute_mixing_statistics` defaults to a documented
symmetrization: it drops self-loops, collapses duplicate and reciprocal arcs to
one unordered edge, and computes `inter_edges / total_edges`. It can also
report the distinct-arc ratio for sensitivity checks. Isolated nodes remain in
per-community node counts but contribute no edge. With no retained edges,
`mu` is `null`/`None`; it is not forced to zero.

### Omega is not a keep probability identity

The existing platform operationalizes a risky candidate's retention as

\[
p_{\mathrm{keep}}=1-r(1-\omega_c),
\]

where \(r\) is its configured risk score and \(c\) is intra or inter according
to the root author's community. This is a stochastic exposure policy. Paper
Eq. (2), in contrast, weights counts of adopted neighbors inside a deterministic
threshold. Therefore:

> **Paper omega and OASIS keep probability are not mathematically equivalent.**

The bridge transfers only the paper-supported *allocation direction*. It then
selects among OASIS keep-probability candidates using observed local outcomes.
Every calibration record is marked `project_adaptation_oasis_observation` and
every selected mapping is marked `project_adaptation_not_paper_equivalence`.

### Allocation and budget

`allocate_cosref_control` accepts measured \(\mu\), a candidate grid, and the
shared project budget

\[
C_{\mathrm{project}}=(1-\omega_{\mathrm{intra}})
                    +(1-\omega_{\mathrm{inter}}).
\]

This L1 budget is a project constraint, not Eq. (5). Each candidate also
reports Eq. (5) for reference. For strong communities the admissible preferred
set has `omega_intra <= omega_inter`; for weak communities it has
`omega_inter <= omega_intra`; the transition band keeps both sides.

### Offline exposure calibration

The calibration scan uses an actual `AdaptiveDiffusionPlatform`, frozen
directed SBM, fixed seed, two high-risk and two benign synthetic seed posts,
and deterministic scripted agents. It records:

- high-risk shown impressions, split by intra/inter root relation;
- successful high-risk reposts and root-aware cascade size;
- community coverage by shown high-risk content;
- benign exposure loss;
- realized expected intervention cost, the sum of `1 - keep_probability`
  over candidates.

Within the direction-consistent shared-budget grid, selection minimizes
observed high-risk exposure, then risk reposts, community coverage, benign
loss, and intervention cost. This lexicographic objective is a project design
choice. Calibration uses a seed disjoint from evaluation seeds.

### Independent five-baseline experiment

`configs/cosref_bridge.json` compares, without any LLM:

1. `no_intervention`;
2. `global_throttle` (uniformly throttles benign and risky candidates);
3. `static_cosref`;
4. the existing `dynamic_cosref` controller;
5. `theory_informed_cosref` selected by the offline calibration above.

For a given topology and evaluation seed, these runs share the generated
network, initial posts, active-user schedule, risk labels, and budget ceiling.
Suppression changes later repost state and therefore later recommendation
candidates, so exact event-by-event pairing is not guaranteed after paths
diverge. The output states this limitation.

## Scope exclusions

This bridge does not reproduce the complete tree-like phase diagrams,
finite-size scaling, SIS extension, SF/RR/real-world experiments, Friendster,
YouTube, Orkut, or a causal platform intervention study. It does not infer
unknown formulas. Its small official-code slice and its OASIS calibration are
engineering checks, not evidence that COSREF has been reproduced in full or
that governance is effective in real populations.
