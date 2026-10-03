from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from adaptive_diffusionguard.evaluation.suite import (
    DecisionRubric,
    classify_rationale_stance,
    load_suite,
)
from adaptive_diffusionguard.evaluation.teacher_eval import (
    EvaluationConfigurationError,
    run_suite,
)

SUITE = Path("configs/teacher_eval_100.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_suite_has_twenty_independent_scenarios_and_five_fixed_batches() -> None:
    suite = load_suite(SUITE)
    assert len(suite["scenarios"]) == 20
    assert len(suite["batches"]) == 5
    assert all(len(batch["scenarios"]) == 4 for batch in suite["batches"])
    assert sum(len(scenario["rubric"]) for scenario in suite["scenarios"]) == 100
    assert len({scenario["scenario_id"] for scenario in suite["scenarios"]}) == 20
    assert len({scenario["random_seed"] for scenario in suite["scenarios"]}) == 20
    assert all(
        len(scenario["communities"]) == 5
        and sorted(scenario["feeds"]) == [str(i) for i in range(5)]
        and sorted(scenario["rubric"]) == [str(i) for i in range(5)]
        for scenario in suite["scenarios"]
    )
    required = {
        "low_risk_only",
        "high_risk_only",
        "mixed_high_low",
        "multiple_high_risk",
        "ambiguous_or_insufficient",
        "credible_source_correction",
        "quote_high_risk_challenge",
        "quote_low_risk_support",
        "intra_community",
        "inter_community",
        "high_risk_cross_community",
        "already_reported",
        "already_reposted_target_or_root",
        "empty_feed",
        "multiple_legal_actions",
        "normal_content_false_report",
        "high_risk_wrong_repost",
        "quote_reasonable",
        "ignore_reasonable",
        "history_dependent_action_mask",
    }
    assert required <= {
        tag for scenario in suite["scenarios"] for tag in scenario["coverage"]
    }
    assert all(
        post["content"].startswith("[SYNTHETIC]")
        for scenario in suite["scenarios"]
        for post in scenario["posts"]
    )


def test_rubric_uses_action_root_and_quote_stance_without_scoring_ignore() -> None:
    rubric = DecisionRubric.from_dict(
        {
            "aligned": [
                {"action": "report", "root_post": "high"},
                {"action": "quote", "root_post": "high", "stance": "challenge"},
            ],
            "acceptable": [{"action": "repost", "root_post": "low"}],
            "concerning": [
                {"action": "report", "root_post": "low"},
                {"action": "repost", "root_post": "high"},
            ],
            "unscored": False,
        }
    )
    assert rubric.classify(action="report", root_post="high", stance="not_applicable") == "aligned"
    assert rubric.classify(action="report", root_post="low", stance="not_applicable") == "concerning"
    assert rubric.classify(action="quote", root_post="high", stance="challenge") == "aligned"
    assert rubric.classify(action="quote", root_post="high", stance="support") == "unscored"
    assert rubric.classify(action="ignore", root_post=None, stance="not_applicable") == "unscored"
    assert classify_rationale_stance("quote", "[stance:challenge] synthetic") == "challenge"
    assert classify_rationale_stance("quote", "No interpretable position") == "unclear"


@pytest.mark.asyncio
async def test_fake_backend_runs_100_native_snapshots_without_leakage(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fake"
    result = await run_suite(SUITE, "fake", output)
    assert result["manifest"]["logical_decisions"] == 100
    assert result["manifest"]["remote_api_calls"] == 0
    assert result["reliability"]["completed_decisions"] == 100
    assert result["reliability"]["physical_remote_attempts"] == 0
    assert result["dataset_validation"] == {
        "action_label_eligible": 100,
        "decision_snapshot_count": 100,
        "failed_snapshot_count": 0,
        "feed_consistency_rate": 1.0,
        "label_leakage_check_passed": True,
        "label_leakage_decision_ids": [],
        "rationale_fact_consistency_review_queue": 100,
        "rationale_training_eligible": 0,
        "secret_scan_passed": True,
        "succeeded_snapshot_count": 100,
        "teacher_candidate_count": 100,
        "trace_binding_rate": 1.0,
        "unique_decision_id_count": 100,
    }
    rows = _jsonl(output / "per_decision.jsonl")
    assert len(rows) == len({row["decision_id"] for row in rows}) == 100
    assert {row["action"] for row in rows} == {"ignore", "repost", "quote", "report"}
    assert len(_jsonl(output / "rationale_review_queue.jsonl")) == 100

    # The root relation, not the direct quote/repost row, supplies risk.
    related = [
        row
        for row in rows
        if row.get("target_logical_post") in {"challenge", "support", "correction", "shared_low"}
    ]
    assert related
    assert all(row["root_logical_post"] != row["target_logical_post"] for row in related)

    # Historical actions change only the relevant user's live legal mask.
    already_reported = next(
        row
        for row in rows
        if row["scenario_id"] == "scenario-12-already_reported" and row["user_id"] == 0
    )
    assert not any(choice.startswith("report:") for choice in already_reported["legal_choice_ids"])
    already_reposted = next(
        row
        for row in rows
        if row["scenario_id"] == "scenario-13-already_reposted_root" and row["user_id"] == 0
    )
    assert not any(choice.startswith("repost:") for choice in already_reposted["legal_choice_ids"])

    db_paths = sorted(output.glob("batches/*/scenarios/*.db"))
    assert len(db_paths) == 20
    run_ids: set[str] = set()
    for database in db_paths:
        with sqlite3.connect(database) as connection:
            snapshots = connection.execute(
                "SELECT decision_id, feed_json, behavior_history_json, "
                "action_trace_rowid, status FROM diffusionguard_decision_snapshot "
                "ORDER BY decision_sequence"
            ).fetchall()
            assert len(snapshots) == 5
            for decision_id, feed_json, history_json, trace_rowid, status in snapshots:
                assert status == "succeeded"
                assert json.loads(feed_json).get("posts") is not None
                assert all(
                    int(token.split(":", 1)[0].removeprefix("trace-")) < trace_rowid
                    for token in json.loads(history_json)
                )
                run_ids.add(decision_id.split(":t", 1)[0])
    assert len(run_ids) == 20


@pytest.mark.asyncio
async def test_local_baselines_are_legal_reproducible_and_batch_resumable(
    tmp_path: Path,
) -> None:
    random_one = tmp_path / "random-one"
    random_two = tmp_path / "random-two"
    deterministic = tmp_path / "deterministic"
    await run_suite(SUITE, "random_legal", random_one)
    await run_suite(SUITE, "random_legal", random_two)
    await run_suite(SUITE, "deterministic_risk_rule", deterministic)

    assert _file_hashes(random_one) == _file_hashes(random_two)
    for output in (random_one, deterministic):
        rows = _jsonl(output / "per_decision.jsonl")
        assert len(rows) == 100
        assert all(row["selected_choice_id"] in row["legal_choice_ids"] for row in rows)
        manifest = _json(output / "manifest.json")
        cache_paths = [batch["response_cache_path"] for batch in manifest["batches"]]
        assert len(set(cache_paths)) == 5
        assert all(batch["logical_decisions"] == 20 for batch in manifest["batches"])
        assert manifest["remote_api_calls"] == 0

    before = _file_hashes(random_one)
    await run_suite(SUITE, "random_legal", random_one, resume=True)
    assert _file_hashes(random_one) == before


@pytest.mark.asyncio
async def test_fake_outputs_are_byte_reproducible_and_groq_is_offline_disabled(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    await run_suite(SUITE, "fake", first)
    await run_suite(SUITE, "fake", second)
    assert _file_hashes(first) == _file_hashes(second)
    with pytest.raises(EvaluationConfigurationError, match="explicit --batch"):
        await run_suite(SUITE, "groq", tmp_path / "groq-must-not-exist")
    assert not (tmp_path / "groq-must-not-exist").exists()
