from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_diffusionguard.theory.allocation import (
    allocate_cosref_control,
    paper_exponential_cost,
    project_l1_cost,
)
from adaptive_diffusionguard.theory.calibration import (
    CalibrationObservation,
    select_calibrated_keep_probabilities,
)
from adaptive_diffusionguard.theory.experiment import run_scenario
from adaptive_diffusionguard.theory.mixing import compute_mixing_statistics
from adaptive_diffusionguard.theory.reference_validation import (
    summarize_reference_run,
)


def test_mixing_manual_network_and_isolated_node() -> None:
    communities = {0: "a", 1: "a", 2: "b", 3: "b", 4: "c"}
    stats = compute_mixing_statistics(
        [(0, 1), (1, 0), (0, 2), (2, 0), (2, 3), (4, 4)],
        communities,
    )
    assert stats.intra_edge_count == 2
    assert stats.inter_edge_count == 1
    assert stats.mu == pytest.approx(1 / 3)
    assert stats.duplicate_edges_ignored == 2
    assert stats.self_loops_ignored == 1
    assert stats.isolated_node_ids == (4,)


@pytest.mark.parametrize(
    ("edges", "expected_mu"),
    [([(0, 1), (2, 3)], 0.0), ([(0, 2), (1, 3)], 1.0), ([], None)],
)
def test_mixing_boundaries(
    edges: list[tuple[int, int]], expected_mu: float | None
) -> None:
    communities = {0: "a", 1: "a", 2: "b", 3: "b"}
    assert compute_mixing_statistics(edges, communities).mu == expected_mu


def test_directed_arc_adaptation_is_explicit() -> None:
    communities = {0: "a", 1: "a", 2: "b"}
    symmetrized = compute_mixing_statistics(
        [(0, 1), (1, 0), (0, 2)], communities
    )
    arcs = compute_mixing_statistics(
        [(0, 1), (1, 0), (0, 2)],
        communities,
        directed_adaptation="arcs",
    )
    assert symmetrized.total_edge_count == 2
    assert arcs.total_edge_count == 3
    assert symmetrized.mu == pytest.approx(0.5)
    assert arcs.mu == pytest.approx(1 / 3)


def test_mixing_rejects_missing_labels_and_invalid_mode() -> None:
    with pytest.raises(ValueError, match="missing community"):
        compute_mixing_statistics([(0, 1)], {0: "a"})
    with pytest.raises(ValueError, match="directed_adaptation"):
        compute_mixing_statistics([], {}, directed_adaptation="invalid")  # type: ignore[arg-type]


def test_mixing_accepts_one_pass_edge_iterables() -> None:
    edges = ((source, target) for source, target in [(0, 1)])
    stats = compute_mixing_statistics(edges, {0: "a", 1: "b"})
    assert stats.mu == 1.0


def test_paper_cost_and_project_budget_boundaries() -> None:
    assert paper_exponential_cost(1, 1) == pytest.approx(0)
    assert paper_exponential_cost(0, 0) == pytest.approx(1)
    assert project_l1_cost(0.6, 0.6) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        paper_exponential_cost(-0.1, 1)


def test_allocation_direction_transition_band_budget_and_determinism() -> None:
    grid = [(0.2, 1.0), (0.6, 0.6), (1.0, 0.2), (0.0, 0.0)]
    strong = allocate_cosref_control(0.2, 0.8, grid, tolerance=0.05)
    weak = allocate_cosref_control(0.8, 0.8, grid, tolerance=0.05)
    transition = allocate_cosref_control(0.52, 0.8, grid, tolerance=0.05)
    assert strong.direction == "intra"
    assert weak.direction == "inter"
    assert transition.direction == "balanced"
    assert strong.candidates[0].omega_intra <= strong.candidates[0].omega_inter
    assert weak.candidates[0].omega_inter <= weak.candidates[0].omega_intra
    assert all(item.project_budget_cost <= 0.8 for item in strong.candidates)
    assert strong == allocate_cosref_control(0.2, 0.8, grid, tolerance=0.05)


def _observation(
    omega: tuple[float, float], high_risk: int, provenance: str = "project_adaptation_oasis_observation"
) -> CalibrationObservation:
    return CalibrationObservation(
        omega_intra=omega[0],
        omega_inter=omega[1],
        seed=1,
        high_risk_exposures=high_risk,
        intra_high_risk_exposures=high_risk,
        inter_high_risk_exposures=0,
        successful_risk_reposts=1,
        cascade_size=3,
        community_coverage=0.5,
        benign_exposure_loss=0.0,
        intervention_cost=2.0,
        provenance=provenance,
    )


def test_calibration_is_project_adaptation_not_paper_result() -> None:
    allocation = allocate_cosref_control(
        0.2, 0.8, [(0.2, 1.0), (0.4, 0.8), (1.0, 0.2)]
    )
    selection = select_calibrated_keep_probabilities(
        [_observation((0.2, 1.0), 3), _observation((0.4, 0.8), 5)], allocation
    )
    assert (selection.omega_intra, selection.omega_inter) == (0.2, 1.0)
    assert selection.provenance == "project_adaptation_not_paper_equivalence"
    with pytest.raises(ValueError, match="project adaptation"):
        select_calibrated_keep_probabilities(
            [_observation((0.2, 1.0), 3, "paper_result")], allocation
        )


@pytest.mark.asyncio
async def test_theory_baseline_uses_real_oasis_and_no_llm(tmp_path: Path) -> None:
    config = {
        "project_budget": 0.8,
        "simulation": {
            "nodes": 8,
            "timesteps": 1,
            "active_users_per_step": 4,
            "recommendation_count": 2,
            "following_count": 1,
            "recommendation_buffer": 4,
            "controller_window": 1,
            "risk_score": 0.9,
        },
        "global_keep_probability": 0.6,
        "static_omega_intra": 0.6,
        "static_omega_inter": 0.6,
        "dynamic_controller": {
            "interval": 1,
            "step_size": 0.2,
            "risk_target": 0.25,
            "cross_community_target": 0.3,
            "report_target": 0.1,
            "benign_loss_limit": 0.1,
        },
    }
    result = await run_scenario(
        condition={"id": "test", "p_intra": 0.5, "p_inter": 0.1},
        config=config,
        baseline="theory_informed_cosref",
        seed=17,
        output_dir=tmp_path / "run",
        theory_omega=(0.4, 0.8),
    )
    assert result.llm_calls == 0
    assert Path(result.database).exists()
    assert result.candidate_impressions > 0
    persisted = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert persisted["provenance"] == "project_adaptation_oasis_experiment"
    repeated = await run_scenario(
        condition={"id": "test", "p_intra": 0.5, "p_inter": 0.1},
        config=config,
        baseline="theory_informed_cosref",
        seed=17,
        output_dir=tmp_path / "repeated",
        theory_omega=(0.4, 0.8),
    )
    first_metrics = result.as_dict()
    repeated_metrics = repeated.as_dict()
    first_metrics.pop("database")
    repeated_metrics.pop("database")
    assert first_metrics == repeated_metrics


def test_reference_summary_detects_expected_direction(tmp_path: Path) -> None:
    raw = tmp_path / "raw.tsv"
    raw.write_text(
        "mu\tomega_intra\tomega_inter\tsample\trho_A\trho_B\n"
        "0.2\t0.0\t0.8\t0\t0.3\t0.1\n"
        "0.2\t0.8\t0.0\t0\t0.9\t0.9\n"
        "0.8\t0.8\t0.0\t0\t0.3\t0.1\n"
        "0.8\t0.0\t0.8\t0\t0.9\t0.9\n",
        encoding="utf-8",
    )
    patch = tmp_path / "instrumentation.patch"
    patch.write_text("fixed seed\n", encoding="utf-8")
    summary = summarize_reference_run(raw, tmp_path / "out", patch)
    assert all(item["priority_observed"] for item in summary["optima"])
