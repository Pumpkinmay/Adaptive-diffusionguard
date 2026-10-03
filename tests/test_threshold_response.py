from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_diffusionguard.theory.threshold_experiment import (
    ThresholdPilot,
    ThresholdScenarioResult,
    run_threshold_scenario,
    select_cost_matched_static,
    validate_threshold_config,
)
from adaptive_diffusionguard.theory.threshold_response import (
    ThresholdResponseEngine,
    allocate_strict_oasis_keep,
    simulate_paper_native,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cosref_threshold_response_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _line_engine(*, threshold: float, exposure_gate: bool = False) -> ThresholdResponseEngine:
    return ThresholdResponseEngine(
        communities={0: "a", 1: "a", 2: "b"},
        contact_edges=[(0, 1), (1, 2)],
        initial_adopters={10: [0]},
        threshold=threshold,
        paper_omega_intra=1.0,
        paper_omega_inter=1.0,
        exposure_gate_enabled=exposure_gate,
    )


def test_threshold_uses_strict_greater_than_and_equality_does_not_adopt() -> None:
    equal = _line_engine(threshold=0.5)
    equal.begin_timestep(1)
    decision = equal.evaluate(
        user_id=1,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[0],
    )
    assert decision.paper_native_signal == decision.threshold_boundary == 1.0
    assert decision.threshold_satisfied is False

    above = _line_engine(threshold=0.49)
    above.begin_timestep(1)
    assert above.evaluate(
        user_id=1,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[0],
    ).should_attempt_adoption


def test_synchronous_update_prevents_same_timestep_cascade() -> None:
    engine = _line_engine(threshold=0.0)
    engine.begin_timestep(1)
    first = engine.evaluate(
        user_id=1,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[0],
    )
    second = engine.evaluate(
        user_id=2,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[1],
    )
    assert first.should_attempt_adoption is True
    assert second.should_attempt_adoption is False
    engine.commit(1, [(1, 10)])
    engine.begin_timestep(2)
    assert engine.evaluate(
        user_id=2,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[1],
    ).should_attempt_adoption


def test_adoption_is_irreversible_and_user_root_is_committed_once() -> None:
    engine = _line_engine(threshold=0.0)
    engine.begin_timestep(1)
    assert engine.commit(1, [(1, 10), (1, 10)]) == ((1, 10),)
    engine.begin_timestep(2)
    decision = engine.evaluate(
        user_id=1,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[],
    )
    assert decision.already_adopted is True
    assert decision.should_attempt_adoption is False
    assert engine.commit(2, [(1, 10)]) == ()
    assert engine.adoption_time(1, 10) == 1


def test_intra_inter_neighbour_counts_and_degree_are_explicit() -> None:
    engine = ThresholdResponseEngine(
        communities={0: "a", 1: "a", 2: "b", 3: "b"},
        contact_edges=[(0, 1), (0, 2), (0, 3)],
        initial_adopters={4: [1, 2]},
        threshold=0.1,
        paper_omega_intra=0.5,
        paper_omega_inter=0.25,
        exposure_gate_enabled=False,
    )
    engine.begin_timestep(1)
    decision = engine.evaluate(
        user_id=0,
        root_post_id=4,
        root_author_community="a",
        observable_adopter_ids=[],
    )
    assert decision.intra_adopted_neighbor_count == 1
    assert decision.inter_adopted_neighbor_count == 1
    assert decision.intra_degree == 1
    assert decision.inter_degree == 2
    assert decision.total_degree == 3
    assert decision.paper_native_signal == 0.75


def test_exposure_gate_blocks_otherwise_eligible_adoption() -> None:
    engine = _line_engine(threshold=0.0, exposure_gate=True)
    engine.begin_timestep(1)
    blocked = engine.evaluate(
        user_id=1,
        root_post_id=10,
        root_author_community="a",
        observable_adopter_ids=[],
    )
    assert blocked.paper_native_threshold_satisfied is True
    assert blocked.observable_threshold_satisfied is False
    assert blocked.threshold_met_but_exposure_blocked is True
    assert blocked.should_attempt_adoption is False


def test_paper_native_process_has_multiple_synchronous_generations() -> None:
    result = simulate_paper_native(
        communities={0: "a", 1: "a", 2: "a"},
        contact_edges=[(0, 1), (1, 2)],
        initial_adopters=[0],
        threshold=0.0,
        paper_omega_intra=1.0,
        paper_omega_inter=1.0,
        timesteps=3,
    )
    assert result["final_adoption_fraction"] == 1.0
    assert [row["new_adopters"] for row in result["timesteps"]] == [1, 1, 0]
    assert result["exposure_gate_enabled"] is False


def test_strict_allocator_rejects_symmetric_priority_and_balances_transition() -> None:
    grid = [(0.2, 1.0), (0.4, 0.8), (0.6, 0.6), (0.8, 0.4), (1.0, 0.2)]
    strong = allocate_strict_oasis_keep(
        mu=0.1,
        project_budget=0.8,
        keep_grid=grid,
        tolerance=0.05,
        minimum_strict_gap=0.2,
    )
    weak = allocate_strict_oasis_keep(
        mu=0.9,
        project_budget=0.8,
        keep_grid=grid,
        tolerance=0.05,
        minimum_strict_gap=0.2,
    )
    balanced = allocate_strict_oasis_keep(
        mu=0.5,
        project_budget=0.8,
        keep_grid=grid,
        tolerance=0.05,
        minimum_strict_gap=0.2,
    )
    assert (0.6, 0.6) not in strong.eligible_oasis_keep_pairs
    assert (0.6, 0.6) not in weak.eligible_oasis_keep_pairs
    assert balanced.eligible_oasis_keep_pairs == ((0.6, 0.6),)
    with pytest.raises(ValueError, match="no keep candidate"):
        allocate_strict_oasis_keep(
            mu=0.1,
            project_budget=0.8,
            keep_grid=[(0.6, 0.6)],
            tolerance=0.05,
            minimum_strict_gap=0.2,
        )


def test_config_uses_actual_mu_per_network_and_separated_parameters() -> None:
    config = _config()
    validation = validate_threshold_config(config)
    assert validation["valid"] is True
    assert validation["llm_calls"] == 0
    assert len(validation["network_measurements"]) == 3 * (1 + 5 + 10)
    directions = {
        row["condition"]: row["allocation_direction"]
        for row in validation["network_measurements"]
    }
    assert directions == {
        "strong-community": "intra",
        "moderate-mixing": "balanced",
        "weak-community": "inter",
    }
    response = config["threshold_response"]
    assert response["main_paper_omega_intra"] == 1.0
    assert response["main_paper_omega_inter"] == 1.0
    assert config["paper_omega_equals_oasis_keep"] is False


def test_calibration_and_evaluation_seeds_must_not_overlap() -> None:
    config = _config()
    config["evaluation_seeds"][0] = config["calibration_seeds"][0]
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_threshold_config(config)


def test_realized_cost_matching_reports_tolerance() -> None:
    matched = select_cost_matched_static(
        target_cost=100.0,
        candidate_mean_costs={(0.5, 0.5): 96.0, (0.6, 0.6): 108.0},
        maximum_relative_error=0.05,
    )
    assert matched["selected_static_keep"] == [0.5, 0.5]
    assert matched["relative_error"] == pytest.approx(0.04)
    assert matched["within_preregistered_tolerance"] is True


def test_resume_rejects_config_hash_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"config_sha256": "different"}), encoding="utf-8"
    )
    pilot = ThresholdPilot(CONFIG_PATH, output, resume=True)
    with pytest.raises(ValueError, match="config hash mismatch"):
        pilot._initialize_output({})


def _fake_scenario_result(output: Path) -> ThresholdScenarioResult:
    output.mkdir(parents=True, exist_ok=False)
    database = output / "simulation.db"
    database.write_bytes(b"sqlite placeholder")
    (output / "threshold_decisions.jsonl").write_text("", encoding="utf-8")
    (output / "timestep_adoption.jsonl").write_text("", encoding="utf-8")
    return ThresholdScenarioResult(
        condition="strong-community",
        strategy="calibration",
        seed=44001,
        measured_mu=0.1,
        allocation_direction="intra",
        oasis_keep_intra=0.2,
        oasis_keep_inter=1.0,
        paper_omega_intra=1.0,
        paper_omega_inter=1.0,
        threshold=0.1,
        initial_adoption_density=0.17,
        initial_risk_adopters=10,
        final_risk_adopters=12,
        final_risk_adoption_fraction=0.2,
        successful_risk_reposts=2,
        risk_cascade_size=12,
        maximum_risk_post_depth=2,
        risk_adoption_generations=2,
        intra_risk_adoptions=10,
        inter_risk_adoptions=2,
        community_coverage=1.0,
        threshold_met_but_exposure_blocked=3,
        below_threshold_count=4,
        candidate_impressions=20,
        shown_impressions=15,
        suppressed_impressions=5,
        high_risk_exposures=8,
        benign_exposure_loss=0.0,
        parameter_l1_cost=0.8,
        realized_intervention_cost=5.0,
        realized_intervention_cost_per_candidate=0.25,
        llm_calls=0,
        database=str(database),
        audit_path=str(output / "threshold_decisions.jsonl"),
        timestep_path=str(output / "timestep_adoption.jsonl"),
    )


@pytest.mark.asyncio
async def test_checkpoint_resume_skips_completed_and_quarantines_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pilot"
    pilot = ThresholdPilot(CONFIG_PATH, output)
    pilot._initialize_output({"within_limit": True})
    condition = pilot.config["conditions"][0]
    calls = 0

    async def fake_run(**kwargs: object) -> ThresholdScenarioResult:
        nonlocal calls
        calls += 1
        return _fake_scenario_result(kwargs["output_dir"])

    monkeypatch.setattr(
        "adaptive_diffusionguard.theory.threshold_experiment.run_threshold_scenario",
        fake_run,
    )
    arguments = {
        "phase": "calibration",
        "condition": condition,
        "label": "keep-0.2-1.0",
        "strategy": "calibration",
        "seed": 44001,
        "keep_pair": (0.2, 1.0),
    }
    first = await pilot._run_one(**arguments)
    resumed = await pilot._run_one(**arguments)
    assert calls == 1
    assert first["llm_calls"] == 0
    assert resumed["resumed_from_checkpoint"] is True

    complete = (
        output
        / "raw"
        / "calibration"
        / "strong-community"
        / "keep-0.2-1.0"
        / "44001"
        / "complete.json"
    )
    complete.unlink()
    rebuilt = await pilot._run_one(**arguments)
    assert calls == 2
    assert rebuilt["status"] == "completed"
    assert len(list((output / "quarantine").rglob("simulation.db"))) == 1


def test_local_paper_native_grid_agrees_with_verified_direction() -> None:
    pilot = ThresholdPilot(CONFIG_PATH, Path("unused"))
    result = pilot._paper_native_check()
    assert result["status"] == "passed"
    assert {row["observed_direction"] for row in result["optima"]} == {
        "intra",
        "inter",
    }
    assert result["llm_calls"] == 0


@pytest.mark.asyncio
async def test_real_oasis_threshold_run_separates_keep_and_paper_omega(
    tmp_path: Path,
) -> None:
    config = _config()
    config["simulation"] = {
        **config["simulation"],
        "nodes": 8,
        "timesteps": 2,
        "recommendation_count": 8,
        "following_count": 8,
        "recommendation_buffer": 16,
    }
    config["threshold_response"] = {
        **config["threshold_response"],
        "initial_adoption_density": 0.25,
        "threshold": 0.0,
    }
    result = await run_threshold_scenario(
        condition={"id": "integration", "p_intra": 0.7, "p_inter": 0.2},
        config=config,
        strategy="calibration",
        seed=91,
        output_dir=tmp_path / "run",
        keep_pair=(0.2, 1.0),
    )
    assert result.llm_calls == 0
    assert (result.paper_omega_intra, result.paper_omega_inter) == (1.0, 1.0)
    assert (result.oasis_keep_intra, result.oasis_keep_inter) == (0.2, 1.0)
    audits = [
        json.loads(line)
        for line in (tmp_path / "run" / "threshold_decisions.jsonl").read_text().splitlines()
    ]
    assert audits
    successful = [row for row in audits if row["dispatcher_success"]]
    assert len({(row["user_id"], row["root_post_id"]) for row in successful}) == len(
        successful
    )
    paired = await run_threshold_scenario(
        condition={"id": "integration", "p_intra": 0.7, "p_inter": 0.2},
        config=config,
        strategy="static_l1",
        seed=91,
        output_dir=tmp_path / "paired",
        keep_pair=(0.2, 1.0),
    )
    first_metrics = result.as_dict()
    paired_metrics = paired.as_dict()
    for field in ("strategy", "database", "audit_path", "timestep_path"):
        first_metrics.pop(field)
        paired_metrics.pop(field)
    assert first_metrics == paired_metrics
