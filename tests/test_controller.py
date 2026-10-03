import pytest

from adaptive_diffusionguard.governance.controller import RuleBasedController
from adaptive_diffusionguard.metrics.state import GovernanceState


def test_controller_bounds_and_budget() -> None:
    controller = RuleBasedController(
        interval=1,
        step_size=0.5,
        risk_target=0.1,
        cross_community_target=0.1,
        report_target=0.1,
        benign_loss_limit=0.2,
        intervention_budget=0.2,
    )
    state = GovernanceState(
        risk_adoption_rate=0.9,
        recent_reposts=20,
        community_coverage_ratio=1.0,
        cross_community_exposure_ratio=0.9,
        report_rate=0.8,
        benign_exposure_loss=0.0,
        remaining_budget=0.2,
    )
    action = controller.update(1, state, 1.0, 1.0)
    assert 0 <= action.omega_intra <= 1
    assert 0 <= action.omega_inter <= 1
    assert action.expected_cost == pytest.approx(0.2)


def test_controller_does_not_update_between_intervals() -> None:
    controller = RuleBasedController(2, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0)
    action = controller.update(1, GovernanceState(remaining_budget=1.0), 0.6, 0.3)
    assert (action.omega_intra, action.omega_inter) == (0.6, 0.3)
