# COSREF threshold-response bridge v1

## Verified paper mechanism

For the paper's undirected contact graph, each node has an irreversible binary
adoption state. For a susceptible node of total degree `k`, adopted
within-community neighbour count `m_intra`, and adopted between-community
neighbour count `m_inter`, paper Eq. (2) adopts exactly when

```text
paper_omega_intra * m_intra + paper_omega_inter * m_inter > theta * k
```

The comparison is strictly greater-than: equality does not adopt. The main
text specifies synchronous updates, so v1 freezes the complete adoption state
at the beginning of a timestep and commits successful new adoptions only after
all users have evaluated that snapshot. Adoption is irreversible. The official
v1.0.0 C++ implementation confirms the weighted inequality and strict
boundary, although its FIFO implementation reaches monotone closure
asynchronously; this bridge follows the main-text synchronous definition.

Initial adopters use the official-code convention: `floor(rho_0 * N)` distinct
nodes sampled from the first community. The OASIS root author is one member of
that sampled set, and the remaining initial adopters are materialized as
initial reposts. They are not counted as response-generated reposts.

## Directed OASIS adaptation

The paper degree is the number of unique neighbours in an undirected contact
network. OASIS follow arcs are therefore symmetrized, reciprocal/duplicate
arcs collapse to one contact, self-loops are excluded, and each contact is
stored in both follow directions. Total, intra-, and inter-community degree
are computed on that same contact set. This is a project adaptation, not a
paper definition for directed social media graphs.

Adoption state is keyed by `(user_id, root_post_id)`. Repost and quote chains
are resolved to the root. A user can adopt a root only once. The OASIS 0.2.5
repost implementation stores reposts against the root, so database post depth
remains one even when synchronous adoption occurs across several timesteps;
the experiment separately records temporal adoption generations.

## Exposure-gated main path

The main experiment uses this single control path:

```text
oasis_keep_intra/inter
  -> OASIS decides which root-related posts are shown
  -> shown posts reveal a subset of frozen adopted contact neighbours
  -> fixed paper threshold weights decide adoption
  -> deterministic OASIS repost dispatcher
```

Only the direct author of a shown root/repost counts as an observable adopted
neighbour, and only if that author is both a contact neighbour and adopted in
the frozen state. Duplicate posts from one neighbour count once. The full
paper-native neighbour signal is also computed for audit. A decision is marked
`threshold_met_but_exposure_blocked` when the full frozen-neighbour signal
passes Eq. (2) but the visible-neighbour signal does not.

The main path fixes `paper_omega_intra=paper_omega_inter=1`. Intervention is
applied only through `oasis_keep_intra/inter`; paper omega is not multiplied by
the same control a second time. Every decision is marked
`project_adaptation_observable_neighbour_threshold`.

## Strict project allocation

Every generated network is measured separately. With tolerance `delta` and
minimum gap `g`, eligible OASIS keep candidates obey:

- `mu < 0.5-delta`: `keep_inter - keep_intra >= g`;
- `mu > 0.5+delta`: `keep_intra - keep_inter >= g`;
- otherwise: `keep_intra == keep_inter`.

All theory candidates also obey the preregistered project L1 budget. A
symmetric candidate is never treated as directional, and no-candidate cases
fail explicitly. This is a strict project allocator carrying the paper's
directional prior; it is not a paper formula or an omega equivalence.

## Paper-native diagnostic

The small diagnostic disables exposure gating and scans paper omega directly
under the synchronous threshold engine. It applies the official plotting
cutoff only as a diagnostic boundary and compares the minimum paper-cost grid
direction with the already archived official oracle. It is not part of the
main OASIS result and is not a reproduction of the full phase diagram.
