# COSREF + LLM fixed-choice semantic protocol v2.1

## Scope

Protocol `fixed-choice-semantic-v2.1` is an additive, versioned protocol. It
does not change the reproducible v1 or v2 execution paths. Its purpose is to
detect a numeric-choice binding failure observed in the real v2 micro-pilot:

`joint-v2-strong-community-static_cosref-61001:t000001:u000031:d000002`

That record executed `report:2`, while its rationale explicitly described
reporting high-risk post 3. Post 2 was the low-risk weather post. The original
run and its artifacts remain unchanged.

## Fixed provider schema

The provider response has five required fields:

```json
{
  "choice_index": 3,
  "action_type": "report",
  "target_post_id": 3,
  "quote_text": "",
  "rationale": "..."
}
```

`action_type` has the fixed enum `ignore`, `report`, `repost`, `quote`. The
schema has no dynamic choice enum, union, nullable field, tool, or
`tool_choice`. The deterministic, ordered legal-action list is part of the
messages and the cache key. `ignore` uses target 0. Other actions use a visible
direct post ID. Only `quote` may have non-empty `quote_text`.

## Dispatch boundary

The local gateway validates the index range, the indexed action type, the
indexed direct post ID, the quote-text contract, the canonical root, root-level
report/repost deduplication, and current OASIS legality. Rationale text is not
parsed to change or authorize an action. It is retained only for later factual
auditing.

A semantic mismatch produces no dispatch, action trace, or succeeded snapshot.
At most one correction request is permitted. It reuses the frozen
DecisionSnapshot, Feed, action order, response schema, and original messages,
adding only this fixed instruction:

> choice_index、action_type和target_post_id必须共同描述同一个列出的合法选项

The correction path does not refresh the Feed and therefore cannot create a
second refresh trace or impression batch. A second mismatch makes the logical
decision fail explicitly; it is never converted to `ignore`.

## Accounting

Attempt-level semantic metrics are kept separate from provider, schema, and
dispatcher failures:

- `semantic_consistency_failure_count`
- `choice_action_mismatch_count`
- `choice_target_mismatch_count`
- `quote_text_contract_failure_count`
- `semantic_correction_attempt_count`
- `semantic_correction_success_count`
- `semantic_correction_failure_count`

Final logical failures use a single mutually exclusive category. Provider
errors are not dispatcher errors; semantic binding errors are not provider or
Pydantic schema errors.

## Counterfactual v2 audit boundary

The read-only v2 audit can verify the known rationale-to-executed-target
conflict. If a v2.1 response had explicitly returned `report` and target 3
with the old index that selected `report:2`, v2.1 would reject it before
dispatch as a target mismatch. The old v2 response did not contain the new
`action_type`, `target_post_id`, or `quote_text` fields, so strict v2.1
consistency for all twenty historical responses is `not_reconstructable`.
This audit does not claim that v2.1 has already changed or validated real model
behavior.

## Experiment boundaries

The v2.1 Fake suite creates no teacher/training samples and performs no remote
requests. The planned semantic real micro-pilot contains 12 logical decisions
and permits at most 24 physical requests. A plan is read-only and does not load
`.env`, create a model, database, cache, or output directory.
