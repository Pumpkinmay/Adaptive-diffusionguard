from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from adaptive_diffusionguard.experiments import cosref_llm_joint as v1
from adaptive_diffusionguard.experiments import cosref_llm_joint_v2_1 as v21
from adaptive_diffusionguard.llm.runtime import LLMRuntime
from adaptive_diffusionguard.llm.structured_actions import ActionMask, DecisionSnapshot
from scripts.audit_joint_v2_1_counterfactual import _tree_digest, audit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosref_llm_joint_v2_1.json"
REAL_V2 = ROOT / "runs/cosref-llm-joint-v2-groq-micro"


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            original_post_id INTEGER,
            quote_content TEXT
        );
        CREATE TABLE report (user_id INTEGER NOT NULL, post_id INTEGER NOT NULL);
        INSERT INTO post VALUES (1, 10, NULL, NULL);
        INSERT INTO post VALUES (2, 11, NULL, NULL);
        INSERT INTO post VALUES (3, 40, NULL, NULL);
        INSERT INTO post VALUES (6, 42, 1, NULL);
        INSERT INTO post VALUES (7, 12, 1, 'corrective quote');
        """
    )
    return connection


def _snapshot(tmp_path: Path) -> tuple[DecisionSnapshot, v21.FakeSemanticModel]:
    connection = _connection()
    builder = v21.SemanticActionMaskBuilder(connection)
    feed = {
        "posts": [
            {"post_id": 1, "content": "Unverified synthetic high-risk claim."},
            {"post_id": 2, "content": "Verified synthetic benign update."},
            {"post_id": 3, "content": "Unverified synthetic high-risk claim."},
        ]
    }
    mask = builder.build(0, {1, 2, 3})
    messages = v21.build_semantic_messages(
        profile="Synthetic verifier who checks risky claims before sharing.",
        feed=feed,
        mask=mask,
        builder=builder,
    )
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed=feed,
        visible_post_ids={1, 2, 3},
        mask=mask,
        response_format=v21.semantic_response_format(),
        messages=messages,
        state_identifier="fixed-v2-1",
    )
    connection.close()
    return snapshot, v21.FakeSemanticModel(tmp_path / "v21.sqlite3")


def _parsed(
    index: int,
    action: str,
    target: int,
    quote_text: str = "",
) -> v21.SemanticActionResponse:
    return v21.SemanticActionResponse.model_validate(
        {
            "choice_index": index,
            "action_type": action,
            "target_post_id": target,
            "quote_text": quote_text,
            "rationale": "Auditable synthetic reason.",
        }
    )


def test_fixed_semantic_schema_has_only_fixed_action_enum_and_no_tools() -> None:
    response_format = v21.semantic_response_format()
    schema = response_format["json_schema"]["schema"]
    assert response_format["json_schema"]["strict"] is True
    assert schema["properties"]["choice_index"] == {"type": "integer"}
    assert schema["properties"]["action_type"]["enum"] == [
        "ignore",
        "report",
        "repost",
        "quote",
    ]
    assert "enum" not in schema["properties"]["choice_index"]
    assert "tools" not in response_format
    assert "tool_choice" not in response_format


@pytest.mark.parametrize(
    ("choice", "parsed"),
    [
        ("ignore", _parsed(0, "ignore", 0)),
        ("report:1", _parsed(1, "report", 1)),
        ("repost:2", _parsed(2, "repost", 2)),
        ("quote:3", _parsed(3, "quote", 3, "corrective text")),
    ],
)
def test_four_actions_have_consistent_semantic_binding(
    choice: str, parsed: v21.SemanticActionResponse
) -> None:
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed={"posts": []},
        visible_post_ids=set(),
        mask=ActionMask(("ignore", "report:1", "repost:2", "quote:3")),
        response_format=v21.semantic_response_format(),
        messages=[],
        state_identifier="x",
    )
    assert v21.validate_semantic_binding(snapshot, parsed) == choice


@pytest.mark.parametrize(
    ("parsed", "error"),
    [
        (_parsed(1, "repost", 1), v21.ChoiceActionMismatch),
        (_parsed(1, "report", 3), v21.ChoiceTargetMismatch),
        (_parsed(0, "ignore", 1), v21.ChoiceTargetMismatch),
        (_parsed(1, "report", 0), v21.ChoiceTargetMismatch),
        (_parsed(3, "quote", 3, ""), v21.QuoteTextContractFailure),
        (_parsed(1, "report", 1, "unexpected"), v21.QuoteTextContractFailure),
    ],
)
def test_semantic_mismatches_are_rejected(
    parsed: v21.SemanticActionResponse,
    error: type[BaseException],
) -> None:
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed={"posts": []},
        visible_post_ids=set(),
        mask=ActionMask(("ignore", "report:1", "repost:2", "quote:3")),
        response_format=v21.semantic_response_format(),
        messages=[],
        state_identifier="x",
    )
    with pytest.raises(error):
        v21.validate_semantic_binding(snapshot, parsed)


@pytest.mark.asyncio
async def test_semantic_correction_reuses_frozen_snapshot(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)
    metrics = v21.SemanticAuditMetrics()
    result = await v21.request_semantic_with_correction(model, snapshot, metrics)
    assert result.retried is True
    assert result.semantic_correction is True
    assert result.messages[:-1] == snapshot.messages_value()
    assert result.messages[-1]["content"] == v21.SEMANTIC_CORRECTION_PROMPT
    assert result.response_format == snapshot.response_format_value()
    assert model.local_attempts == 2
    assert metrics.semantic_consistency_failure_count == 1
    assert metrics.choice_target_mismatch_count == 1
    assert metrics.semantic_correction_attempt_count == 1


@pytest.mark.asyncio
async def test_second_semantic_mismatch_stops(tmp_path: Path) -> None:
    snapshot, _ = _snapshot(tmp_path)
    model = v21.FakeSemanticModel(
        tmp_path / "always.sqlite3", failure_mode="always_semantic_mismatch"
    )
    metrics = v21.SemanticAuditMetrics()
    with pytest.raises(v21.ChoiceTargetMismatch):
        await v21.request_semantic_with_correction(model, snapshot, metrics)
    assert model.local_attempts == 2
    assert metrics.semantic_consistency_failure_count == 2
    assert metrics.semantic_correction_attempt_count == 1
    assert metrics.semantic_correction_failure_count == 1


@pytest.mark.asyncio
async def test_provider_schema_and_semantic_failures_are_separate(
    tmp_path: Path,
) -> None:
    snapshot, _ = _snapshot(tmp_path)
    model = v21.FakeSemanticModel(
        tmp_path / "provider.sqlite3", failure_mode="once_json_validate"
    )
    metrics = v21.SemanticAuditMetrics()
    result = await v21.request_semantic_with_correction(model, snapshot, metrics)
    assert result.retried is True
    assert result.semantic_correction is False
    assert model.runtime.stats.json_validate_failed_count == 1
    assert metrics.semantic_consistency_failure_count == 0
    assert metrics.semantic_correction_attempt_count == 0


def test_root_report_and_repost_deduplication_remains_active() -> None:
    connection = _connection()
    builder = v21.SemanticActionMaskBuilder(connection)
    connection.execute("INSERT INTO report VALUES (0, 6)")
    connection.execute("INSERT INTO post VALUES (8, 0, 7, NULL)")
    mask = builder.build(0, {1, 6, 7})
    assert not any(choice.startswith("report:") for choice in mask.choices)
    assert not any(choice.startswith("repost:") for choice in mask.choices)
    assert sum(choice.startswith("quote:") for choice in mask.choices) == 3
    connection.close()


def test_v2_1_cache_key_is_isolated_from_v2() -> None:
    messages = [{"role": "system", "content": v21.PROTOCOL_VERSION}]
    common = {
        "provider": "groq",
        "model": "fake",
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": messages,
    }
    v21_key = LLMRuntime.cache_key(
        **common,
        model_config={"protocol_version": v21.PROTOCOL_VERSION},
        response_format=v21.semantic_response_format(),
    )
    v2_key = LLMRuntime.cache_key(
        **common,
        model_config={"protocol_version": "fixed-choice-index-v2"},
        response_format={"type": "json_schema", "version": "v2"},
    )
    assert v21_key != v2_key


def test_semantic_micro_plan_is_read_only(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    plan = v21.build_plan(
        CONFIG, "groq", output, semantic_micro_pilot=True
    )
    assert not output.exists()
    assert plan["logical_decisions"] == 12
    assert plan["maximum_physical_requests_total"] == 24
    assert plan["protocol_version"] == v21.PROTOCOL_VERSION
    assert plan["response_schema_dynamic"] is False
    assert plan["tools_sent"] is False
    assert plan["tool_choice_sent"] is False
    assert {unit["condition_id"] for unit in plan["units"]} == {
        "strong-community",
        "moderate-mixing",
        "weak-community",
    }


def test_groq_confirmation_fails_before_env_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("environment must not be read")

    monkeypatch.setattr(v21, "load_dotenv", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v21",
            "--config",
            str(CONFIG),
            "--backend",
            "groq",
            "--output",
            str(tmp_path / "out"),
            "--semantic-micro-pilot",
        ],
    )
    with pytest.raises(SystemExit, match="confirm-remote-run"):
        v21.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_full_fake_v2_1_run_completes_90(tmp_path: Path) -> None:
    output = tmp_path / "v2-1-fake"
    result = await v21.run_joint_v2_1(CONFIG, "fake", output)
    reliability = result["reliability"]
    integrity = result["integrity"]
    assert result["summary"]["status"] == "success"
    assert result["summary"]["completed_decisions"] == 90
    assert reliability["complete_units"] == 9
    assert reliability["physical_backend_attempts"] == 180
    assert reliability["semantic_consistency_failure_count"] == 90
    assert reliability["choice_target_mismatch_count"] == 90
    assert reliability["semantic_correction_attempt_count"] == 90
    assert reliability["semantic_correction_success_count"] == 90
    assert reliability["semantic_correction_failure_count"] == 0
    assert reliability["dispatcher_failure_count"] == 0
    assert reliability["physical_remote_attempts"] == 0
    assert integrity["trace_binding_rate"] == 1.0
    assert integrity["semantic_dispatch_guard_passed"] is True
    assert integrity["root_action_deduplication_passed"] is True
    assert integrity["threshold_shadow_non_mutating"] is True
    assert integrity["refresh_trace_count"] == 90
    assert integrity["semantic_correction_extra_refresh_count"] == 0
    assert integrity["semantic_correction_added_impression_batch"] is False
    assert integrity["teacher_or_training_samples_created"] == 0
    assert len(list(output.glob("units/*/*/complete.json"))) == 9


@pytest.mark.asyncio
async def test_final_semantic_failure_never_reaches_trace_or_success_snapshot(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fake_backend"]["failure_mode"] = "always_semantic_mismatch"
    network = v1._generate_network(config, config["network"]["conditions"][0])
    directory = tmp_path / "failed-unit"
    decisions, summary, runtime = await v21._run_unit(
        config=config,
        config_path=CONFIG,
        network=network,
        strategy="no_intervention",
        backend="fake",
        directory=directory,
        selected={(1, 0)},
        llm_settings=None,
        model_factory=lambda _: None,
    )
    assert summary["completed_decisions"] == 0
    assert decisions[0]["status"] == "failed"
    assert decisions[0]["failure_category"] == "semantic_consistency_failure"
    assert runtime["semantic_correction_failure_count"] == 1
    with sqlite3.connect(directory / "experiment.db") as connection:
        status, trace_rowid = connection.execute(
            "SELECT status, action_trace_rowid "
            "FROM diffusionguard_decision_snapshot"
        ).fetchone()
        refreshes = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE action = 'refresh'"
        ).fetchone()[0]
        user_actions = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE user_id = 0 "
            "AND action IN ('report_post', 'repost', 'quote_post', 'do_nothing')"
        ).fetchone()[0]
    assert (status, trace_rowid) == ("failed", None)
    assert refreshes == 1
    assert user_actions == 0


def test_real_v2_counterfactual_audit_is_read_only_and_bounded() -> None:
    expected = "338bd89be966f43ba609cfaf19b96a9a08b8f6ba0f51dde404448f601b90cd5d"
    before = _tree_digest(REAL_V2)
    result = audit(REAL_V2)
    after = _tree_digest(REAL_V2)
    assert before == after == expected
    assert result["decision_count"] == 20
    assert result["v2_index_to_execution_consistent_count"] == 20
    assert result["v2_1_directly_reconstructable_count"] == 0
    assert result["v2_1_not_reconstructable_count"] == 20
    assert result["known_record_verified"] is True
    assert result["known_record_counterfactual_rejected"] is True
    assert result["remote_api_calls"] == 0
    assert result["secret_scan_passed"] is True


def test_v1_frozen_tree_digest_unchanged() -> None:
    root = ROOT / "runs/cosref-llm-joint-v1-groq"
    digest = hashlib.sha256()
    excluded = {"corrected_accounting_audit.json", "v1_role_quality_audit.json"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert digest.hexdigest() == (
        "61ce077a5bbecad50ae5de58ceb3ebab757194c81acfb321d2bdd9ffd16f2828"
    )
