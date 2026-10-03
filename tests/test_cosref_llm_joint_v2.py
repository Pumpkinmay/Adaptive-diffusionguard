from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from adaptive_diffusionguard.experiments import cosref_llm_joint_v2 as v2
from adaptive_diffusionguard.llm.runtime import LLMRuntime
from adaptive_diffusionguard.llm.structured_actions import ActionMask, DecisionSnapshot
from scripts.audit_cosref_llm_joint_v1 import EXPECTED_HASHES, audit

CONFIG = Path("configs/cosref_llm_joint_v2.json")
V1_OUTPUT = Path("runs/cosref-llm-joint-v1-groq")


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
        CREATE TABLE report (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL
        );
        INSERT INTO post VALUES (1, 10, NULL, NULL);
        INSERT INTO post VALUES (6, 42, 1, NULL);
        INSERT INTO post VALUES (7, 12, 1, 'corrective quote');
        """
    )
    return connection


def _snapshot(
    tmp_path: Path,
    *,
    choices: tuple[str, ...] = ("ignore", "report:1"),
) -> tuple[DecisionSnapshot, v2.FakeIndexModel]:
    connection = _connection()
    builder = v2.RootAwareActionMaskBuilder(connection)
    mask = ActionMask(choices)
    feed = {
        "success": True,
        "posts": [
            {
                "post_id": 1,
                "user_id": 10,
                "content": "Unverified synthetic high-risk claim.",
            }
        ],
    }
    messages = v2.build_index_messages(
        profile="Synthetic verifier who checks risky claims before sharing.",
        feed=feed,
        mask=mask,
        builder=builder,
        agent_role="behavior_simulator",
    )
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed=feed,
        visible_post_ids={1},
        mask=mask,
        response_format=v2.fixed_index_response_format(),
        messages=messages,
        state_identifier="fixed",
    )
    model = v2.FakeIndexModel(tmp_path / "v2-cache.sqlite3")
    connection.close()
    return snapshot, model


def test_role_contracts_are_explicit_and_distinct() -> None:
    behavior = v2.ROLE_CONTRACTS["behavior_simulator"]
    safety = v2.ROLE_CONTRACTS["safety_policy_agent"]
    assert behavior["safety_adverse_is_not_automatically_reasoning_error"] is True
    assert "reverse_action_rules" not in behavior
    assert "reverse_action_rules" in safety
    assert behavior["oracle_risk_labels_in_prompt"] is False
    assert safety["oracle_risk_labels_in_prompt"] is False


def test_unverified_is_not_misread_as_verified_low_risk() -> None:
    rationale = "The selected post contains an unverified synthetic claim."
    assert v2._rationale_fact_consistent("report", 0.9, rationale) is True


def test_fixed_schema_has_integer_index_and_no_dynamic_enum_or_tools() -> None:
    response_format = v2.fixed_index_response_format()
    schema = response_format["json_schema"]["schema"]
    assert response_format["json_schema"]["strict"] is True
    assert schema["properties"]["choice_index"] == {"type": "integer"}
    assert "enum" not in json.dumps(response_format)
    assert "tools" not in response_format
    assert "tool_choice" not in response_format


def test_ordered_actions_live_in_messages_and_exclude_oracle_treatment() -> None:
    connection = _connection()
    builder = v2.RootAwareActionMaskBuilder(connection)
    mask = builder.build(0, {1, 6, 7})
    messages = v2.build_index_messages(
        profile="Synthetic profile",
        feed={"posts": []},
        mask=mask,
        builder=builder,
        agent_role="behavior_simulator",
    )
    payload = json.loads(messages[1]["content"])
    assert [item["choice_index"] for item in payload["ordered_legal_actions"]] == list(
        range(len(mask.choices))
    )
    assert payload["protocol_version"] == v2.PROTOCOL_VERSION
    serialized = json.dumps(messages).lower()
    assert "risk_score" not in serialized
    assert "static_cosref" not in serialized
    assert "theory_informed_cosref" not in serialized
    connection.close()


def test_choice_index_range_and_strict_integer_validation() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"choice_index": "1", "rationale": "reason"}
                    )
                },
            }
        ]
    }
    with pytest.raises(v2.LocalSchemaFailure):
        v2.parse_index_response(response)
    mask = ActionMask(("ignore", "report:1"))
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed={"posts": []},
        visible_post_ids=set(),
        mask=mask,
        response_format=v2.fixed_index_response_format(),
        messages=[],
        state_identifier="x",
    )
    with pytest.raises(v2.InvalidChoiceIndex):
        v2.choice_from_index(snapshot, 2)
    with pytest.raises(v2.InvalidChoiceIndex):
        v2.choice_from_index(snapshot, -1)


@pytest.mark.asyncio
async def test_same_snapshot_correction_succeeds_once(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)
    result = await v2.request_fixed_index_with_correction(model, snapshot)
    assert result.retried is True
    assert result.messages[:-1] == snapshot.messages_value()
    assert result.messages[-1]["content"] == v2.V2_CORRECTION_PROMPT
    assert result.response_format == snapshot.response_format_value()
    assert model.local_attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("always_json_validate", Exception),
        ("out_of_range", v2.InvalidChoiceIndex),
        ("non_integer", v2.LocalSchemaFailure),
    ],
)
async def test_second_invalid_attempt_stops_without_loop(
    tmp_path: Path, mode: str, error: type[BaseException]
) -> None:
    snapshot, _ = _snapshot(tmp_path)
    model = v2.FakeIndexModel(
        tmp_path / f"{mode}.sqlite3", failure_mode=mode
    )
    with pytest.raises(error):
        await v2.request_fixed_index_with_correction(model, snapshot)
    assert model.local_attempts == 2
    assert model.runtime.stats.structured_retry_attempt_count == 1
    assert model.runtime.stats.structured_retry_failure_count == 1


@pytest.mark.parametrize("reported_post", [1, 6, 7])
def test_report_deduplicates_every_derivative_of_same_root(
    reported_post: int,
) -> None:
    connection = _connection()
    builder = v2.RootAwareActionMaskBuilder(connection)
    connection.execute(
        "INSERT INTO report (user_id, post_id) VALUES (0, ?)", (reported_post,)
    )
    mask = builder.build(0, {1, 6, 7})
    assert not any(choice.startswith("report:") for choice in mask.choices)
    assert [choice for choice in mask.choices if choice.startswith("quote:")] == [
        "quote:1",
        "quote:6",
        "quote:7",
    ]
    connection.close()


@pytest.mark.parametrize("reposted_parent", [1, 6, 7])
def test_repost_deduplicates_every_derivative_of_same_root(
    reposted_parent: int,
) -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO post VALUES (8, 0, ?, NULL)", (reposted_parent,)
    )
    builder = v2.RootAwareActionMaskBuilder(connection)
    mask = builder.build(0, {1, 6, 7})
    assert not any(choice.startswith("repost:") for choice in mask.choices)
    assert any(choice.startswith("quote:") for choice in mask.choices)
    connection.close()


def test_quote_history_does_not_suppress_repeated_quote_semantics() -> None:
    connection = _connection()
    connection.execute("INSERT INTO post VALUES (8, 0, 1, 'prior quote')")
    builder = v2.RootAwareActionMaskBuilder(connection)
    mask = builder.build(0, {1, 6, 7})
    assert [choice for choice in mask.choices if choice.startswith("quote:")] == [
        "quote:1",
        "quote:6",
        "quote:7",
    ]
    connection.close()


def test_dispatch_precheck_rejects_root_state_change() -> None:
    connection = _connection()
    builder = v2.RootAwareActionMaskBuilder(connection)
    mask = builder.build(0, {1, 6})
    index = mask.choices.index("report:6")
    snapshot = DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed={"posts": []},
        visible_post_ids={1, 6},
        mask=mask,
        response_format=v2.fixed_index_response_format(),
        messages=[],
        state_identifier="before",
    )
    connection.execute("INSERT INTO report (user_id, post_id) VALUES (0, 1)")
    with pytest.raises(v2.CurrentStateChoiceInvalid):
        v2.revalidate_index_choice(snapshot, index, builder)
    connection.close()


def test_cache_key_includes_protocol_and_ordered_action_list() -> None:
    base = [{"role": "system", "content": v2.PROTOCOL_VERSION}]
    first = base + [{"role": "user", "content": "0:ignore,1:report:1"}]
    second = base + [{"role": "user", "content": "0:ignore,1:quote:1"}]
    kwargs = {
        "provider": "groq",
        "model": "fake",
        "temperature": 0.0,
        "max_tokens": 512,
        "model_config": {"protocol_version": v2.PROTOCOL_VERSION},
        "response_format": v2.fixed_index_response_format(),
    }
    assert LLMRuntime.cache_key(messages=first, **kwargs) != LLMRuntime.cache_key(
        messages=second, **kwargs
    )
    v1_key = LLMRuntime.cache_key(
        messages=[{"role": "system", "content": "dynamic-choice-id-v1"}],
        **kwargs,
    )
    assert v1_key != LLMRuntime.cache_key(messages=first, **kwargs)


def test_failure_accounting_is_mutually_exclusive() -> None:
    accounting = v2.FailureAccounting()
    accounting.record_completed()
    accounting.record_failure("provider_json_validate_failed")
    accounting.record_failure("dispatcher_failure")
    result = accounting.to_dict()
    assert result["provider_failure_count"] == 1
    assert result["provider_json_validate_failed_count"] == 1
    assert result["dispatcher_failure_count"] == 1
    assert result["incomplete_decision_count"] == 2
    assert result["unrecovered_logical_error_count"] == 2


def test_micro_plan_is_read_only_and_covers_twenty_decisions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "not-created"
    plan = v2.build_plan(CONFIG, "groq", output, micro_pilot=True)
    assert not output.exists()
    assert plan["logical_decisions"] == 20
    assert plan["maximum_physical_requests_total"] == 40
    assert plan["model_created"] is False
    assert plan["output_created"] is False
    assert {unit["condition_id"] for unit in plan["units"]} == {
        "strong-community",
        "moderate-mixing",
        "weak-community",
    }


def test_groq_requires_micro_and_confirmation_before_env_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("environment must not be read")

    monkeypatch.setattr(v2, "load_dotenv", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v2",
            "--config",
            str(CONFIG),
            "--backend",
            "groq",
            "--output",
            str(tmp_path / "out"),
            "--micro-pilot",
        ],
    )
    with pytest.raises(SystemExit, match="confirm-remote-run"):
        v2.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_full_fake_v2_run_is_complete_and_root_deduplicated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v2-fake"
    result = await v2.run_joint_v2(CONFIG, "fake", output)
    assert result["summary"]["status"] == "success"
    assert result["summary"]["completed_decisions"] == 90
    assert result["reliability"]["complete_units"] == 9
    assert result["reliability"]["physical_backend_attempts"] == 99
    assert result["reliability"]["physical_remote_attempts"] == 0
    assert result["reliability"]["provider_failure_count"] == 0
    assert result["reliability"]["dispatcher_failure_count"] == 0
    assert result["reliability"]["unrecovered_logical_error_count"] == 0
    assert result["integrity"]["trace_binding_rate"] == 1.0
    assert result["integrity"]["feed_consistency_rate"] == 1.0
    assert result["integrity"]["root_action_deduplication_passed"] is True
    assert result["integrity"]["threshold_shadow_non_mutating"] is True
    assert result["integrity"]["secret_scan_passed"] is True
    assert result["manifest"]["teacher_or_training_samples_created"] == 0
    assert len(list(output.glob("units/*/*/complete.json"))) == 9


def test_v1_corrected_audit_preserves_frozen_hashes() -> None:
    before = {name: hashlib.sha256((V1_OUTPUT / name).read_bytes()).hexdigest() for name in EXPECTED_HASHES}
    corrected, roles = audit(V1_OUTPUT)
    after = {name: hashlib.sha256((V1_OUTPUT / name).read_bytes()).hexdigest() for name in EXPECTED_HASHES}
    assert before == after == EXPECTED_HASHES
    assert corrected["provider_failure_count"] == 12
    assert corrected["dispatcher_success_count"] == 78
    assert corrected["dispatcher_failure_count"] == 0
    assert corrected["incomplete_decision_count"] == 12
    assert corrected["unrecovered_logical_error_count"] == 12
    assert len(corrected["root_level_repeated_actions"]) == 3
    assert roles["single_accuracy_reported"] is False
    assert roles["rationale_training_eligible_count"] == 0
