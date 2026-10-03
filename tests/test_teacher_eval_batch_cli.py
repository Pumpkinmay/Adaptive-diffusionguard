from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from adaptive_diffusionguard.evaluation import teacher_eval
from adaptive_diffusionguard.evaluation.teacher_eval import (
    EvaluationConfigurationError,
    ResumeIntegrityError,
    build_plan,
    main,
    run_suite,
)

SUITE = Path("configs/teacher_eval_100.json")
BATCH_SCENARIOS = {
    "scenario-01-low_risk_only",
    "scenario-02-high_risk_only",
    "scenario-03-mixed_risk",
    "scenario-04-multiple_high_risk",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_plan_is_suite_only_and_reports_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.setenv(
        "GROQ_API_KEY", "gsk_" + "fake_plan_credential_123456789"
    )
    monkeypatch.setattr(
        teacher_eval,
        "load_dotenv",
        lambda *args, **kwargs: pytest.fail("plan read an env file"),
    )
    monkeypatch.setattr(
        teacher_eval.LLMSettings,
        "from_env",
        lambda *args, **kwargs: pytest.fail("plan read process credentials"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diffusionguard-teacher-eval",
            "--suite",
            str(SUITE),
            "--backend",
            "groq",
            "--batch",
            "batch-01",
            "--output",
            str(output),
            "--resume",
            "--plan",
        ],
    )
    main()
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "backend": "groq",
        "logical_decisions": 20,
        "maximum_physical_requests": 30,
        "output_path": str(output),
        "resume_requested": True,
        "scenario_ids": sorted(BATCH_SCENARIOS),
        "selected_batch": "batch-01",
    }
    assert not output.exists()
    assert "gsk_" not in json.dumps(plan)


def test_groq_requires_batch_before_env_and_invalid_batch_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "groq-no-batch"
    env_reads = 0

    def fail_env(*args, **kwargs) -> None:
        nonlocal env_reads
        env_reads += 1
        pytest.fail("environment loading must occur after batch validation")

    monkeypatch.setattr(teacher_eval, "load_dotenv", fail_env)
    monkeypatch.setattr(sys, "argv", [
        "diffusionguard-teacher-eval",
        "--suite", str(SUITE),
        "--backend", "groq",
        "--output", str(output),
    ])
    with pytest.raises(SystemExit) as missing:
        main()
    assert missing.value.code != 0
    assert env_reads == 0
    assert not output.exists()

    invalid_output = tmp_path / "invalid"
    with pytest.raises(EvaluationConfigurationError, match="unknown batch"):
        build_plan(
            SUITE, "fake", "batch-01,batch-02", invalid_output, resume=False
        )
    assert not invalid_output.exists()


@pytest.mark.asyncio
async def test_single_batch_runs_only_four_scenarios_and_writes_integrity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "single"
    result = await run_suite(
        SUITE, "fake", output, selected_batch="batch-01"
    )
    manifest = result["manifest"]
    assert result["summary"] == {
        "completed_decisions": 20,
        "exit_code": 0,
        "expected_decisions": 20,
        "logical_decisions": 20,
        "selected_batch": "batch-01",
        "status": "success",
        "status_reasons": [],
    }
    assert manifest["selected_batch"] == "batch-01"
    assert manifest["available_batches"] == [
        "batch-01", "batch-02", "batch-03", "batch-04", "batch-05"
    ]
    assert manifest["logical_decisions"] == 20
    assert manifest["scenario_count"] == 4
    assert manifest["max_physical_requests"] == 30
    assert manifest["partial_suite"] is True
    assert result["reliability"]["logical_decisions"] == 20
    assert result["reliability"]["completed_decisions"] == 20
    assert sum(result["quality"][key] for key in (
        "aligned_count", "acceptable_count", "concerning_count", "unscored_count"
    )) == 20
    assert {row["scenario_id"] for row in _read_jsonl(output / "per_decision.jsonl")} == BATCH_SCENARIOS
    assert not any((output / "batches" / f"batch-0{number}").exists() for number in range(2, 6))

    complete = _read_json(output / "batches" / "batch-01" / "complete.json")
    assert complete["config_sha256"] == manifest["config_sha256"]
    assert complete["backend"] == "fake"
    assert complete["expected_logical_decisions"] == 20
    assert complete["actual_completed_decisions"] == 20
    assert set(complete["scenario_ids"]) == BATCH_SCENARIOS
    assert complete["integrity"]["per_decision"]["rows"] == 20
    assert len(complete["integrity"]["databases"]) == 4


@pytest.mark.asyncio
async def test_local_backend_without_batch_remains_full_suite(tmp_path: Path) -> None:
    result = await run_suite(SUITE, "fake", tmp_path / "full")
    assert result["summary"]["status"] == "success"
    assert result["summary"]["logical_decisions"] == 100
    assert result["manifest"]["partial_suite"] is False
    assert result["manifest"]["selected_batch"] is None
    assert result["manifest"]["scenario_count"] == 20
    assert len(result["manifest"]["batches"]) == 5


@pytest.mark.asyncio
async def test_valid_resume_rebuilds_summary_without_executing_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "resume-complete"
    first = await run_suite(SUITE, "fake", output, selected_batch="batch-01")

    async def forbidden(*args, **kwargs):
        pytest.fail("a validated complete batch executed a decision")

    monkeypatch.setattr(teacher_eval, "_run_scenario", forbidden)
    resumed = await run_suite(
        SUITE,
        "fake",
        output,
        selected_batch="batch-01",
        resume=True,
        model_factory=lambda settings: pytest.fail("resume created a model"),
    )
    assert resumed == first
    assert resumed["manifest"]["remote_api_calls"] == 0


@pytest.mark.asyncio
async def test_incomplete_batch_is_quarantined_without_touching_other_batches(
    tmp_path: Path,
) -> None:
    output = tmp_path / "quarantine"
    await run_suite(SUITE, "fake", output, selected_batch="batch-01")
    (output / "batches" / "batch-01" / "complete.json").unlink()
    other = output / "batches" / "batch-02"
    other.mkdir()
    sentinel = other / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    fixed_time = datetime(2026, 9, 29, 12, 34, 56, tzinfo=UTC)
    result = await run_suite(
        SUITE,
        "fake",
        output,
        selected_batch="batch-01",
        resume=True,
        quarantine_now_factory=lambda: fixed_time,
    )
    quarantine = output / "quarantine" / "batch-01-20260929T123456.000000Z"
    assert quarantine.is_dir()
    assert not (quarantine / "complete.json").exists()
    assert (output / "batches" / "batch-01" / "complete.json").is_file()
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert result["summary"]["completed_decisions"] == 20


@pytest.mark.asyncio
async def test_resume_refuses_changed_suite_hash(tmp_path: Path) -> None:
    output = tmp_path / "hash"
    await run_suite(SUITE, "fake", output, selected_batch="batch-01")
    changed_suite = tmp_path / "changed-suite.json"
    payload = json.loads(SUITE.read_text(encoding="utf-8"))
    payload["description"] += " Changed for resume-integrity test."
    changed_suite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeIntegrityError, match="SHA-256 changed"):
        await run_suite(
            changed_suite,
            "fake",
            output,
            selected_batch="batch-01",
            resume=True,
        )
    assert not (output / "quarantine").exists()


@pytest.mark.asyncio
async def test_partial_failure_is_degraded_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = teacher_eval._select_choice
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic local failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(teacher_eval, "_select_choice", fail_once)
    result = await run_suite(
        SUITE, "fake", tmp_path / "degraded", selected_batch="batch-01"
    )
    assert result["summary"]["status"] == "degraded"
    assert result["summary"]["exit_code"] != 0
    assert result["summary"]["completed_decisions"] == 19
    assert result["dataset_validation"]["teacher_candidate_count"] == 19


def test_cli_exit_code_matches_degraded_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def degraded_result(*args, **kwargs):
        return {"summary": {"exit_code": 2, "status": "degraded"}}

    monkeypatch.setattr(teacher_eval, "run_suite", degraded_result)
    monkeypatch.setattr(sys, "argv", [
        "diffusionguard-teacher-eval",
        "--suite", str(SUITE),
        "--backend", "fake",
        "--batch", "batch-01",
        "--output", str(tmp_path / "degraded-cli"),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
