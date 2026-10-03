"""Read-only source audit for the frozen COSREF + LLM v1 Groq pilot.

The script never reads an environment file or provider payload.  It writes new
derived audit files beside the frozen outputs and verifies the known hashes of
the original summary/configuration artifacts before and after analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_HASHES = {
    "summary.json": "84b4172338263f43b6178d8a5da0fbbaf97caf87f5537d7a07cdbcec4c48c942",
    "reliability_summary.json": "ca7f059e17d9d0129e356edb99b83b77358021381a3605dee90c5c6a797e6170",
    "integrity_audit.json": "acb02e6857fcbb2a19ebbbdb5fccb0af99d06405a556728da5a1521739625276",
    "per_decision.jsonl": "792329b79f27ca82d8af75d777052ae9ae7525dea20e6f84d88a64a2aff514be",
}
SECRET_PATTERN = re.compile(rb"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _original_hashes(output: Path) -> dict[str, str]:
    hashes = {name: _sha(output / name) for name in EXPECTED_HASHES}
    bad = {
        name: {"expected": EXPECTED_HASHES[name], "actual": digest}
        for name, digest in hashes.items()
        if digest != EXPECTED_HASHES[name]
    }
    if bad:
        raise RuntimeError(f"frozen v1 artifact hash mismatch: {sorted(bad)}")
    return hashes


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _behavior_classification(
    profile: str,
    action: str,
    risk: float | None,
    factual_error: bool,
) -> str:
    if factual_error:
        return "factual_rationale_error"
    if profile == "verifier" and action == "report" and risk is not None and risk >= 0.7:
        return "profile_consistent"
    if profile == "benign-sharer" and action == "repost" and risk is not None and risk <= 0.3:
        return "profile_consistent"
    if profile == "contextualizer" and action == "quote" and risk is not None and risk >= 0.7:
        return "profile_consistent"
    if profile == "cautious" and action == "ignore":
        return "profile_consistent"
    if profile == "balanced" and (
        (action == "report" and risk is not None and risk >= 0.7)
        or (action == "repost" and risk is not None and risk <= 0.3)
    ):
        return "profile_consistent"
    if action == "repost" and risk is not None and risk >= 0.7 and profile in {
        "verifier",
        "benign-sharer",
        "balanced",
    }:
        return "profile_inconsistent"
    return "unscored"


def audit(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _original_hashes(output)
    decisions = _read_jsonl(output / "per_decision.jsonl")
    by_id = {str(row["decision_id"]): row for row in decisions}
    units: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    succeeded_shapes: list[tuple[int, int]] = []
    failed_shapes: list[tuple[int, int]] = []
    behavior_counts: Counter[str] = Counter()
    behavior_ids: dict[str, list[str]] = defaultdict(list)
    legal_safety_adverse: list[str] = []
    low_reports: list[str] = []
    high_reposts: list[str] = []
    missed: list[str] = []
    quote_stance: Counter[str] = Counter()
    quote_ids: dict[str, list[str]] = defaultdict(list)
    factual_errors: list[dict[str, Any]] = []
    root_actions: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    runtime_totals: Counter[str] = Counter()

    for summary_path in sorted(output.glob("units/*/*/unit_summary.json")):
        directory = summary_path.parent
        unit_summary = _read_json(summary_path)
        runtime = _read_json(directory / "runtime_metrics.json")
        condition = str(unit_summary["condition_id"])
        strategy = str(unit_summary["strategy"])
        for key in (
            "first_attempt_count",
            "first_attempt_success_count",
            "json_validate_failed_count",
            "structured_retry_attempt_count",
            "structured_retry_success_count",
            "structured_retry_failure_count",
            "unrecovered_error_count",
        ):
            runtime_totals[key] += int(runtime.get(key, 0))
        units.append(
            {
                "condition_id": condition,
                "strategy": strategy,
                "logical_decisions": int(unit_summary["logical_decisions"]),
                "completed_decisions": int(unit_summary["completed_decisions"]),
                "final_provider_failures": int(runtime["unrecovered_error_count"]),
                "structured_retry_attempts": int(
                    runtime["structured_retry_attempt_count"]
                ),
                "structured_retry_successes": int(
                    runtime["structured_retry_success_count"]
                ),
                "structured_retry_failures": int(
                    runtime["structured_retry_failure_count"]
                ),
            }
        )
        connection = sqlite3.connect(directory / "experiment.db")
        connection.row_factory = sqlite3.Row
        posts = {
            int(row["post_id"]): {
                "author": int(row["user_id"]),
                "parent": row["original_post_id"],
            }
            for row in connection.execute(
                "SELECT post_id, user_id, original_post_id FROM post"
            )
        }

        def root(post_id: int, post_map: dict[int, dict[str, Any]] = posts) -> int:
            current = post_id
            seen: set[int] = set()
            while current not in seen:
                seen.add(current)
                parent = post_map[current]["parent"]
                if parent is None:
                    return current
                current = int(parent)
            raise RuntimeError("post cycle in frozen v1 database")

        snapshots = connection.execute(
            "SELECT * FROM diffusionguard_decision_snapshot "
            "ORDER BY decision_sequence, decision_id"
        ).fetchall()
        for snapshot in snapshots:
            decision_id = str(snapshot["decision_id"])
            row = by_id[decision_id]
            legal = json.loads(snapshot["legal_choice_ids_json"])
            feed = json.loads(snapshot["feed_json"]).get("posts", [])
            shape = (len(legal), len(feed))
            if snapshot["status"] == "failed":
                failed_shapes.append(shape)
                failed.append(
                    {
                        "decision_id": decision_id,
                        "condition_id": condition,
                        "strategy": strategy,
                        "timestep": int(snapshot["timestep"]),
                        "user_id": int(snapshot["user_id"]),
                        "stable_error_category": "provider_json_validate_failed",
                        "failure_stage": "provider_strict_schema_validation",
                        "legal_action_count": len(legal),
                        "feed_size": len(feed),
                    }
                )
                continue
            succeeded_shapes.append(shape)
            action = str(row["action"])
            risk_raw = row.get("root_risk_score")
            risk = float(risk_raw) if risk_raw is not None else None
            rationale = str(snapshot["rationale"] or "")
            lower = rationale.lower()
            selected_post = row.get("target_post_id")
            factual_reasons: list[str] = []
            says_high = any(
                token in lower
                for token in ("high-risk", "high‑risk", "unverified", "misinformation")
            )
            says_low = any(
                token in lower for token in ("verified benign", "benign weather")
            )
            if risk is not None and risk <= 0.3 and says_high:
                factual_reasons.append("low_risk_target_described_as_high_risk")
            if risk is not None and risk >= 0.7 and action == "repost" and says_low:
                factual_reasons.append("high_risk_target_described_as_benign")
            mentioned_users = [
                int(value)
                for value in re.findall(r"(?i)\buser[\s\u202f]+(\d+)\b", rationale)
            ]
            if selected_post is not None and mentioned_users:
                author = posts[int(selected_post)]["author"]
                if author not in mentioned_users:
                    factual_reasons.append("selected_post_author_relation_error")
            factual = bool(factual_reasons)
            if factual:
                factual_errors.append(
                    {
                        "decision_id": decision_id,
                        "reasons": sorted(set(factual_reasons)),
                    }
                )
            classification = _behavior_classification(
                str(row["profile_id"]), action, risk, factual
            )
            behavior_counts[classification] += 1
            behavior_ids[classification].append(decision_id)
            if action == "repost" and risk is not None and risk >= 0.7:
                high_reposts.append(decision_id)
                legal_safety_adverse.append(decision_id)
            if action == "report" and risk is not None and risk <= 0.3:
                low_reports.append(decision_id)
            if action == "quote":
                stance = (
                    "corrective_or_questioning_claimed"
                    if any(
                        token in lower
                        for token in ("correct", "question", "unverified", "high-risk", "high‑risk")
                    )
                    else "unclear"
                )
                quote_stance[stance] += 1
                quote_ids[stance].append(decision_id)
            if action == "ignore":
                high_shown = {
                    int(item[0])
                    for item in connection.execute(
                        "SELECT post_id FROM diffusionguard_impression "
                        "WHERE run_id = ? AND user_id = ? AND timestep = ? "
                        "AND shown = 1 AND risk_score >= 0.7",
                        (
                            snapshot["run_id"],
                            int(snapshot["user_id"]),
                            int(snapshot["timestep"]),
                        ),
                    ).fetchall()
                }
                if any(
                    choice.startswith("report:")
                    and int(choice.partition(":")[2]) in high_shown
                    for choice in legal
                ):
                    missed.append(decision_id)
                    legal_safety_adverse.append(decision_id)
            if action in {"report", "repost"} and row.get("root_post_id") is not None:
                root_actions[
                    (
                        str(row["run_id"]),
                        int(row["user_id"]),
                        action,
                        root(int(row["target_post_id"])),
                    )
                ].append(
                    {
                        "decision_id": decision_id,
                        "selected_post_id": int(row["target_post_id"]),
                    }
                )
        connection.close()

    repeated_roots = [
        {
            "run_id": key[0],
            "user_id": key[1],
            "action": key[2],
            "canonical_root_post_id": key[3],
            "decisions": sorted(values, key=lambda item: item["decision_id"]),
        }
        for key, values in sorted(root_actions.items())
        if len(values) > 1
    ]
    corrected = {
        "audit_type": "derived_read_only_v1_corrected_accounting",
        "source_output": str(output),
        "source_hashes_before": before,
        "logical_decision_count": len(decisions),
        "completed_decision_count": sum(
            row.get("status") == "succeeded" for row in decisions
        ),
        "incomplete_decision_count": len(failed),
        "provider_failure_count": len(failed),
        "provider_json_validate_failed_count": len(failed),
        "local_schema_failure_count": 0,
        "invalid_choice_index_count": 0,
        "dispatcher_success_count": sum(
            row.get("status") == "succeeded" for row in decisions
        ),
        "dispatcher_failure_count": 0,
        "unrecovered_logical_error_count": len(failed),
        "retry_accounting": {
            "first_attempt_count": int(runtime_totals["first_attempt_count"]),
            "first_attempt_success_count": int(
                runtime_totals["first_attempt_success_count"]
            ),
            "structured_retry_attempt_count": int(
                runtime_totals["structured_retry_attempt_count"]
            ),
            "structured_retry_success_count": int(
                runtime_totals["structured_retry_success_count"]
            ),
            "structured_retry_failure_count": int(
                runtime_totals["structured_retry_failure_count"]
            ),
            "provider_json_validate_failed_attempt_count": int(
                runtime_totals["json_validate_failed_count"]
            ),
        },
        "failed_decisions": sorted(failed, key=lambda item: item["decision_id"]),
        "shape_association_descriptive_only": {
            "failed": {
                "count": len(failed_shapes),
                "mean_legal_action_count": _mean([item[0] for item in failed_shapes]),
                "mean_feed_size": _mean([item[1] for item in failed_shapes]),
                "legal_action_count_distribution": dict(
                    sorted(Counter(item[0] for item in failed_shapes).items())
                ),
                "feed_size_distribution": dict(
                    sorted(Counter(item[1] for item in failed_shapes).items())
                ),
            },
            "succeeded": {
                "count": len(succeeded_shapes),
                "mean_legal_action_count": _mean(
                    [item[0] for item in succeeded_shapes]
                ),
                "mean_feed_size": _mean([item[1] for item in succeeded_shapes]),
                "legal_action_count_distribution": dict(
                    sorted(Counter(item[0] for item in succeeded_shapes).items())
                ),
                "feed_size_distribution": dict(
                    sorted(Counter(item[1] for item in succeeded_shapes).items())
                ),
            },
            "causal_claim": False,
        },
        "unit_accounting": units,
        "root_level_repeated_actions": repeated_roots,
        "deprecated_v1_fields": {
            "dispatcher_failure_count": {
                "old_value": 12,
                "corrected_value": 0,
                "reason": "v1 computed logical minus completed, not evidenced dispatcher failures",
            },
            "unrecovered_error_count": {
                "old_value": 24,
                "corrected_value": 12,
                "reason": "v1 added the same 12 incomplete decisions to 12 provider failures",
            },
        },
        "sensitive_provider_payload_read": False,
        "remote_api_calls": 0,
    }
    role_audit = {
        "audit_type": "two_role_views_v1_offline",
        "behavior_simulator_view": {
            "classification_counts": {
                name: int(behavior_counts.get(name, 0))
                for name in (
                    "profile_consistent",
                    "profile_inconsistent",
                    "factual_rationale_error",
                    "unscored",
                )
            },
            "decision_ids_by_classification": {
                name: sorted(values) for name, values in sorted(behavior_ids.items())
            },
            "legal_but_safety_adverse_count": len(set(legal_safety_adverse)),
            "legal_but_safety_adverse_decision_ids": sorted(
                set(legal_safety_adverse)
            ),
            "notice": (
                "High-risk repost and ignore are propagation outcomes, not automatically "
                "behavior-model inference errors. Unclear profile cases remain unscored."
            ),
        },
        "safety_policy_view": {
            "assumed_role": "safety_policy_agent",
            "low_risk_report_count": len(low_reports),
            "low_risk_report_decision_ids": sorted(low_reports),
            "high_risk_repost_count": len(high_reposts),
            "high_risk_repost_decision_ids": sorted(high_reposts),
            "missed_intervention_count": len(missed),
            "missed_intervention_decision_ids": sorted(missed),
            "quote_stance_counts": dict(sorted(quote_stance.items())),
            "quote_decision_ids_by_stance": {
                name: sorted(values) for name, values in sorted(quote_ids.items())
            },
            "notice": (
                "Counterfactual safety rubric only; not an ordinary-user behavior score."
            ),
        },
        "factual_rationale_errors": sorted(
            factual_errors, key=lambda item: item["decision_id"]
        ),
        "single_accuracy_reported": False,
        "rationale_training_eligible_count": 0,
        "remote_api_calls": 0,
    }
    after = _original_hashes(output)
    corrected["source_hashes_after"] = after
    corrected["source_hashes_unchanged"] = before == after
    if SECRET_PATTERN.search(
        json.dumps([corrected, role_audit], ensure_ascii=False).encode()
    ):
        raise RuntimeError("credential-like pattern detected in derived audit")
    return corrected, role_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cosref-llm-joint-v1-groq"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrected, role_audit = audit(args.output)
    _write_json(args.output / "corrected_accounting_audit.json", corrected)
    _write_json(args.output / "v1_role_quality_audit.json", role_audit)
    print(
        json.dumps(
            {
                "completed_decision_count": corrected["completed_decision_count"],
                "provider_failure_count": corrected["provider_failure_count"],
                "dispatcher_failure_count": corrected["dispatcher_failure_count"],
                "unrecovered_logical_error_count": corrected[
                    "unrecovered_logical_error_count"
                ],
                "source_hashes_unchanged": corrected["source_hashes_unchanged"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
