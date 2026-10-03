"""State passed to governance controllers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GovernanceState:
    risk_adoption_rate: float = 0.0
    recent_reposts: int = 0
    community_coverage_ratio: float = 0.0
    cross_community_exposure_ratio: float = 0.0
    report_rate: float = 0.0
    benign_exposure_loss: float = 0.0
    cumulative_intervention_cost: float = 0.0
    remaining_budget: float = 0.0

    def __post_init__(self) -> None:
        ratios = (
            self.risk_adoption_rate,
            self.community_coverage_ratio,
            self.cross_community_exposure_ratio,
            self.report_rate,
            self.benign_exposure_loss,
        )
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("state ratios must be in [0, 1]")
        if self.recent_reposts < 0:
            raise ValueError("recent_reposts cannot be negative")
        if self.cumulative_intervention_cost < 0 or self.remaining_budget < 0:
            raise ValueError("budget values cannot be negative")
