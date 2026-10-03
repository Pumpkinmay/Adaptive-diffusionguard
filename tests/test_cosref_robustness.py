from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from adaptive_diffusionguard.theory.experiment import ScenarioResult
from adaptive_diffusionguard.theory.robustness import (
    RobustnessPilot,
    bootstrap_mean_interval,
    paired_differences,
    sha256_file,
    validate_preregistered_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cosref_robustness_pilot_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_preregistered_seeds_are_disjoint_and_mu_conditions_hold() -> None:
    validation = validate_preregistered_config(_config())
    assert validation["valid"] is True
    assert validation["llm_calls"] == 0
    measurements = validation["network_measurements"]
    assert len(measurements) == 3 * (5 + 10 + 1)
    assert {row["phase"] for row in measurements} == {
        "calibration",
        "evaluation",
        "preflight",
    }


def test_overlapping_calibration_and_evaluation_seeds_are_rejected() -> None:
    config = _config()
    config["evaluation_seeds"][0] = config["calibration_seeds"][0]
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_preregistered_config(config)


def test_all_preregistered_baselines_obey_shared_parameter_budget() -> None:
    config = _config()
    validate_preregistered_config(config)
    config["global_keep_probability"] = 0.5
    with pytest.raises(ValueError, match="baseline parameters exceed"):
        validate_preregistered_config(config)


def test_bootstrap_interval_is_fixed_seed_reproducible() -> None:
    first = bootstrap_mean_interval([1.0, 2.0, 4.0, 8.0], resamples=500, seed=9)
    second = bootstrap_mean_interval([1.0, 2.0, 4.0, 8.0], resamples=500, seed=9)
    assert first == second


def test_paired_differences_match_exact_seed() -> None:
    theory = [{"seed": 2, "metric": 4}, {"seed": 1, "metric": 7}]
    baseline = [{"seed": 1, "metric": 3}, {"seed": 2, "metric": 9}]
    assert paired_differences(theory, baseline, "metric") == [
        {"seed": 1, "theory": 7.0, "comparator": 3.0, "difference": 4.0},
        {"seed": 2, "theory": 4.0, "comparator": 9.0, "difference": -5.0},
    ]


def test_resume_rejects_changed_config_hash(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"config_sha256": "not-the-current-config"}), encoding="utf-8"
    )
    pilot = RobustnessPilot(CONFIG_PATH, output, resume=True)
    with pytest.raises(ValueError, match="config hash mismatch"):
        pilot._initialize_output({})


def _fake_result(
    output_dir: Path, condition: str, baseline: str, seed: int
) -> ScenarioResult:
    output_dir.mkdir(parents=True, exist_ok=False)
    database = output_dir / "simulation.db"
    database.write_bytes(b"synthetic sqlite placeholder")
    return ScenarioResult(
        condition=condition,
        baseline=baseline,
        seed=seed,
        measured_mu=0.1,
        omega_intra_initial=0.2,
        omega_inter_initial=1.0,
        omega_intra_final=0.2,
        omega_inter_final=1.0,
        candidate_impressions=10,
        shown_impressions=8,
        suppressed_impressions=2,
        high_risk_candidates=5,
        high_risk_exposures=3,
        intra_high_risk_exposures=2,
        inter_high_risk_exposures=1,
        benign_candidates=5,
        benign_exposure_loss=0.0,
        successful_reposts=1,
        successful_risk_reposts=1,
        risk_cascade_size=3,
        community_coverage=0.5,
        realized_intervention_cost=2.0,
        nominal_project_cost_initial=0.8,
        nominal_paper_cost_initial=0.5,
        reports=1,
        llm_calls=0,
        database=str(database),
    )


@pytest.mark.asyncio
async def test_valid_checkpoint_resume_does_not_repeat_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pilot"
    pilot = RobustnessPilot(CONFIG_PATH, output)
    pilot._initialize_output({"within_limit": True})
    calls = 0

    async def fake_run_scenario(**kwargs: object) -> ScenarioResult:
        nonlocal calls
        calls += 1
        return _fake_result(
            kwargs["output_dir"],
            kwargs["condition"]["id"],
            kwargs["baseline"],
            kwargs["seed"],
        )

    monkeypatch.setattr(
        "adaptive_diffusionguard.theory.robustness.run_scenario", fake_run_scenario
    )
    condition = pilot.config["conditions"][0]
    first = await pilot._run_one(
        phase="calibration",
        condition=condition,
        label="omega-0.2-1.0",
        seed=42001,
        baseline="calibration",
        theory_omega=(0.2, 1.0),
    )
    second = await pilot._run_one(
        phase="calibration",
        condition=condition,
        label="omega-0.2-1.0",
        seed=42001,
        baseline="calibration",
        theory_omega=(0.2, 1.0),
    )
    assert calls == 1
    assert first["llm_calls"] == 0
    assert second["resumed_from_checkpoint"] is True


@pytest.mark.asyncio
async def test_incomplete_run_is_quarantined_and_not_counted_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pilot"
    pilot = RobustnessPilot(CONFIG_PATH, output)
    pilot._initialize_output({"within_limit": True})
    condition = pilot.config["conditions"][0]
    final = (
        output
        / "raw"
        / "calibration"
        / "strong-community"
        / "omega-0.2-1.0"
        / "42001"
    )
    final.mkdir(parents=True)
    (final / "summary.json").write_text("{}\n", encoding="utf-8")
    (final / "simulation.db").write_bytes(b"incomplete")
    assert pilot._complete_is_valid(final) is False
    calls = 0

    async def fake_run_scenario(**kwargs: object) -> ScenarioResult:
        nonlocal calls
        calls += 1
        return _fake_result(
            kwargs["output_dir"],
            kwargs["condition"]["id"],
            kwargs["baseline"],
            kwargs["seed"],
        )

    monkeypatch.setattr(
        "adaptive_diffusionguard.theory.robustness.run_scenario", fake_run_scenario
    )
    result = await pilot._run_one(
        phase="calibration",
        condition=condition,
        label="omega-0.2-1.0",
        seed=42001,
        baseline="calibration",
        theory_omega=(0.2, 1.0),
    )
    assert calls == 1
    assert result["status"] == "completed"
    assert pilot._complete_is_valid(final) is True
    assert len(list((output / "quarantine").rglob("simulation.db"))) == 1


def test_config_file_hash_is_stable_and_nonempty() -> None:
    copied = copy.deepcopy(_config())
    assert copied["experiment_id"] == "cosref-robustness-pilot-v1"
    assert len(sha256_file(CONFIG_PATH)) == 64
