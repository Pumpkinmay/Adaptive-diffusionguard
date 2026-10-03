from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from adaptive_diffusionguard.experiments import cosref_llm_joint as joint
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    DecisionSnapshot,
    build_response_format,
)
from adaptive_diffusionguard.llm.structured_retry import (
    request_with_structured_correction,
)

CONFIG = Path("configs/cosref_llm_joint_v1.json")


def _messages(profile: str, posts: list[dict[str, object]], choices: list[str]):
    return [
        {"role": "system", "content": "fixed"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "profile": profile,
                    "visible_feed": posts,
                    "legal_choice_ids": choices,
                }
            ),
        },
    ]


def _snapshot(choices: tuple[str, ...]) -> DecisionSnapshot:
    mask = ActionMask(choices)
    return DecisionSnapshot.capture(
        user_id=0,
        timestep=1,
        feed={"success": False, "posts": []},
        visible_post_ids=set(),
        mask=mask,
        response_format=build_response_format(mask),
        messages=_messages("Synthetic cautious user", [], list(choices)),
        state_identifier="fixed-state",
    )


def test_plan_has_nine_units_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    plan = joint.build_plan(CONFIG, "groq", output, resume=False)
    assert not output.exists()
    assert plan["experiment_units"] == 9
    assert plan["logical_decisions"] == 90
    assert plan["maximum_physical_requests_per_unit"] == 20
    assert plan["maximum_physical_requests_total"] == 180
    assert plan["request_interval_seconds"] == 12.0
    assert len({row["unit_id"] for row in plan["units"]}) == 9
    measured = {
        row["condition_id"]: row["measured_mu"] for row in plan["units"]
    }
    assert measured["strong-community"] < 0.2
    assert 0.4 <= measured["moderate-mixing"] <= 0.6
    assert measured["weak-community"] > 0.8


def test_theory_policy_uses_actual_mu_not_condition_name() -> None:
    config = joint._load_config(CONFIG)
    conditions = config["network"]["conditions"]
    records = {}
    for condition in conditions:
        network = joint._generate_network(config, condition)
        _, record = joint._policy_for(
            "theory_informed_cosref", config, network
        )
        records[network.condition_id] = record
    assert records["strong-community"]["allocation_direction"] == "intra"
    assert records["strong-community"]["oasis_keep_intra"] == 0.2
    assert records["moderate-mixing"]["allocation_direction"] == "balanced"
    assert records["weak-community"]["allocation_direction"] == "inter"
    assert all(record["paper_omega_intra"] is None for record in records.values())


@pytest.mark.parametrize(
    ("profile", "posts", "choices", "expected"),
    [
        (
            "Synthetic verifier",
            [{"post_id": 1, "content": "Unverified synthetic high-risk claim"}],
            ["ignore", "report:1"],
            "report:1",
        ),
        (
            "Synthetic user sharing clearly verified information",
            [{"post_id": 2, "content": "Verified synthetic benign update"}],
            ["ignore", "repost:2"],
            "repost:2",
        ),
        (
            "Synthetic corrective context user",
            [
                {"post_id": 1, "content": "Unverified synthetic high-risk claim"},
                {"post_id": 2, "content": "Verified synthetic benign update"},
            ],
            ["ignore", "quote:1", "repost:2"],
            "quote:1",
        ),
        ("Synthetic cautious user", [], ["ignore"], "ignore"),
    ],
)
def test_fake_backend_covers_feed_action_cases(
    profile: str,
    posts: list[dict[str, object]],
    choices: list[str],
    expected: str,
) -> None:
    choice, rationale = joint.FakeJointModel._choose(
        _messages(profile, posts, choices), choices
    )
    assert choice == expected
    assert rationale


@pytest.mark.asyncio
async def test_fake_structured_retry_succeeds_once(tmp_path: Path) -> None:
    model = joint.FakeJointModel(
        tmp_path / "cache.sqlite3", inject_first_validation_failure=True
    )
    result = await request_with_structured_correction(model, _snapshot(("ignore",)))
    assert result.retried is True
    assert model.local_attempts == 2
    assert model.runtime.stats.json_validate_failed_count == 1
    assert model.runtime.stats.structured_retry_attempt_count == 1


@pytest.mark.asyncio
async def test_fake_final_failure_is_bounded(tmp_path: Path) -> None:
    model = joint.FakeJointModel(
        tmp_path / "cache.sqlite3",
        inject_first_validation_failure=False,
        always_fail_validation=True,
    )
    with pytest.raises(Exception, match="fake strict validation rejection"):
        await request_with_structured_correction(model, _snapshot(("ignore",)))
    assert model.local_attempts == 2
    assert model.runtime.stats.structured_retry_attempt_count == 1
    assert model.runtime.stats.structured_retry_failure_count == 1
    assert model.runtime.stats.unrecovered_error_count == 1


@pytest.mark.asyncio
async def test_final_failure_does_not_bind_trace_or_training_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = joint._load_config(CONFIG)
    network = joint._generate_network(config, config["network"]["conditions"][0])

    def failing_model(backend, config, cache_path, llm_settings, model_factory):
        del backend, config, llm_settings, model_factory
        return joint.FakeJointModel(
            cache_path,
            inject_first_validation_failure=False,
            always_fail_validation=True,
        )

    monkeypatch.setattr(joint, "_model_for_unit", failing_model)
    decisions, summary, _ = await joint._run_unit(
        config=config,
        config_path=CONFIG,
        network=network,
        strategy="no_intervention",
        backend="fake",
        directory=tmp_path / "failed-unit",
        llm_settings=None,
        model_factory=lambda _: None,
    )
    assert summary["completed_decisions"] == 0
    assert all(row["status"] == "failed" for row in decisions)
    assert all(row["training_eligible"] is False for row in decisions)
    with sqlite3.connect(tmp_path / "failed-unit" / "experiment.db") as connection:
        statuses = connection.execute(
            "SELECT status, action_trace_rowid FROM diffusionguard_decision_snapshot"
        ).fetchall()
    assert len(statuses) == 10
    assert all(status == "failed" and trace is None for status, trace in statuses)


@pytest.mark.asyncio
async def test_full_fake_run_and_resume_are_complete_and_nonduplicating(
    tmp_path: Path,
) -> None:
    output = tmp_path / "joint"
    first = await joint.run_joint(CONFIG, "fake", output)
    assert first["summary"]["status"] == "success"
    assert first["summary"]["completed_decisions"] == 90
    assert first["reliability"]["physical_backend_attempts"] == 99
    assert first["reliability"]["physical_remote_attempts"] == 0
    assert first["reliability"]["structured_retry_success_count"] == 9
    assert first["integrity"]["decision_snapshot_count"] == 90
    assert first["integrity"]["trace_binding_rate"] == 1.0
    assert first["integrity"]["feed_consistency_rate"] == 1.0
    assert first["integrity"]["policy_impression_consistency_rate"] == 1.0
    assert first["integrity"]["threshold_shadow_non_mutating"] is True
    assert first["integrity"]["initial_condition_pairing_passed"] is True
    assert first["integrity"]["treatment_isolation_passed"] is True
    assert first["integrity"]["secret_scan_passed"] is True
    assert first["integrity"]["expected_exposure_differences_observed"] is True
    assert first["manifest"]["oracle_synthetic_risk_labels"] is True
    assert not list(output.rglob("*teacher*.jsonl"))

    complete_paths = sorted(output.glob("units/*/*/complete.json"))
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in complete_paths}
    resumed = await joint.run_joint(CONFIG, "fake", output, resume=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in complete_paths}
    assert resumed["reliability"]["resumed_units"] == 9
    assert before == after
    assert resumed["integrity"]["decision_snapshot_count"] == 90
    assert resumed["integrity"]["duplicate_decision_id_count"] == 0


def test_groq_requires_confirmation_before_environment_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "remote"

    def forbidden(*args, **kwargs):
        raise AssertionError("environment must not be read")

    monkeypatch.setattr(joint, "load_dotenv", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "joint",
            "--config",
            str(CONFIG),
            "--backend",
            "groq",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit, match="requires --confirm-remote-run"):
        joint.main()
    assert not output.exists()


@pytest.mark.asyncio
async def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    copied = tmp_path / "config.json"
    copied.write_bytes(CONFIG.read_bytes())
    output = tmp_path / "joint"
    await joint.run_joint(copied, "fake", output)
    config = json.loads(copied.read_text())
    config["version"] = 2
    copied.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(joint.ResumeIntegrityError, match="configuration SHA-256"):
        await joint.run_joint(copied, "fake", output, resume=True)


@pytest.mark.asyncio
async def test_incomplete_unit_is_quarantined_and_rebuilt_from_boundary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "joint"
    await joint.run_joint(CONFIG, "fake", output)
    untouched = (
        output / "units" / "weak-community" / "static_cosref" / "complete.json"
    )
    untouched_hash = hashlib.sha256(untouched.read_bytes()).hexdigest()
    incomplete = output / "units" / "strong-community" / "static_cosref"
    (incomplete / "complete.json").unlink()
    resumed = await joint.run_joint(CONFIG, "fake", output, resume=True)
    assert resumed["summary"]["status"] == "success"
    assert resumed["reliability"]["resumed_units"] == 8
    assert (incomplete / "complete.json").is_file()
    assert hashlib.sha256(untouched.read_bytes()).hexdigest() == untouched_hash
    quarantined = list(
        (output / "quarantine").glob("strong-community--static_cosref-*")
    )
    assert len(quarantined) == 1
    assert not (quarantined[0] / "complete.json").exists()
