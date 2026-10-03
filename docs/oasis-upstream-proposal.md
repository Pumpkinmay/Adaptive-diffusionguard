# OASIS upstream proposal draft (not submitted)

## Issue draft

**Title:** Add an optional pluggable recommendation/exposure callback

OASIS currently computes a `rec` table and then performs additional feed
assembly in `Platform.refresh()`: it samples recommendation rows and, for
non-Reddit modes, merges posts from followed users. This makes it difficult
for experiments to apply one general policy to every item that can be exposed
without replacing `refresh()`.

Would maintainers be open to a small, backward-compatible callback such as:

```python
RecommendationPolicy.filter(
    *, user_id: int, candidates: list[FeedCandidate], context: FeedContext
) -> list[FeedCandidate]
```

`FeedCandidate` could minimally contain `post_id`, `source`, and an optional
base score. The default implementation would return candidates unchanged, so
existing behavior and databases would be unaffected. The hook would run after
recommendation and followed-user candidates are combined, but before post
records are fetched and the refresh trace is written.

This proposal deliberately excludes experiment-specific risk scores, COSREF,
community labels, and intervention logging. Those belong in downstream
projects. The generic value is enabling ranking, safety, diversity, and audit
experiments to control the actual final candidate set.

Questions for maintainers:

1. Is `Platform.refresh()` the desired stable extension point?
2. Should callbacks be sync-only initially, or support async policies?
3. Should `update_rec_table()` optionally preserve base scores in a new table,
   or should scores remain an opaque callback concern?

## PR draft

**Title:** feat: add no-op recommendation policy hook to Platform refresh

Proposed scope:

- Add small `FeedCandidate` and `RecommendationPolicy` protocol types.
- Add a default pass-through policy.
- Convert recommendation and following rows to candidates in `refresh()`.
- Invoke the policy once on the merged, deduplicated set.
- Preserve current selection, ordering, trace, and response behavior under the
  default policy.
- Add focused tests proving the callback sees both candidate sources and that
  the default output is unchanged.
- Add one documentation example with a generic allow-list policy.

Out of scope: COSREF, misinformation classifiers, community assignment,
budgets, controller logic, and experiment-specific schemas.

No issue or PR has been created from this draft.
