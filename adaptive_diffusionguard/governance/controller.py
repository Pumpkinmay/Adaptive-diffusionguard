"""Interpretable dynamic controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from adaptive_diffusionguard.metrics.state import GovernanceState


@dataclass(frozen=True, slots=True)
class ControlAction:
    omega_intra: float
    omega_inter: float
    expected_cost: float
    reason: str


class Controller(ABC):
    """Shared interface for rule-based, MPC, or future controllers."""

    @abstractmethod
    def update(
        self,
        timestep: int,
        state: GovernanceState,
        current_omega_intra: float,
        current_omega_inter: float,
    ) -> ControlAction:
        """Return bounded parameters subject to the shared budget."""


class RuleBasedController(Controller):
    """Lower keep rates when measured diffusion pressure exceeds targets."""

    def __init__(
        self,
        interval: int,
        step_size: float,
        risk_target: float,
        cross_community_target: float,
        report_target: float,
        benign_loss_limit: float,
        intervention_budget: float,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if step_size < 0:
            raise ValueError("step_size cannot be negative")
        targets = {
            "risk_target": risk_target,
            "cross_community_target": cross_community_target,
            "report_target": report_target,
            "benign_loss_limit": benign_loss_limit,
        }
        for name, value in targets.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.interval = interval
        self.step_size = min(float(step_size), 1.0)
        self.risk_target = risk_target
        self.cross_community_target = cross_community_target
        self.report_target = report_target
        self.benign_loss_limit = benign_loss_limit
        self.intervention_budget = max(float(intervention_budget), 0.0)

    @staticmethod
    def _clip(value: float) -> float:
        return min(1.0, max(0.0, value))

    def update(
        self,
        timestep: int,
        state: GovernanceState,
        current_omega_intra: float,
        current_omega_inter: float,
    ) -> ControlAction:
        intra = self._clip(current_omega_intra)
        inter = self._clip(current_omega_inter)
        if timestep % self.interval:
            return ControlAction(intra, inter, 0.0, "between control intervals")

        remaining = min(
            state.remaining_budget,
            max(0.0, self.intervention_budget - state.cumulative_intervention_cost),
        )
        if remaining <= 0:
            return ControlAction(intra, inter, 0.0, "budget exhausted")

        pressure = max(
            state.risk_adoption_rate - self.risk_target,
            state.report_rate - self.report_target,
            0.0,
        )
        cross_pressure = max(
            state.cross_community_exposure_ratio - self.cross_community_target,
            0.0,
        )
        propagation_signal = min(1.0, state.recent_reposts / 10.0)
        coverage_signal = state.community_coverage_ratio
        relax = self.step_size if state.benign_exposure_loss > self.benign_loss_limit else 0.0

        intra_delta = self.step_size * min(1.0, pressure + 0.5 * propagation_signal)
        inter_delta = self.step_size * min(
            1.0, pressure + cross_pressure + 0.5 * coverage_signal
        )
        proposed_intra = self._clip(intra - intra_delta + relax)
        proposed_inter = self._clip(inter - inter_delta + relax)
        raw_cost = (intra - proposed_intra) + (inter - proposed_inter)
        if raw_cost > remaining and raw_cost > 0:
            scale = remaining / raw_cost
            proposed_intra = self._clip(intra + (proposed_intra - intra) * scale)
            proposed_inter = self._clip(inter + (proposed_inter - inter) * scale)
            raw_cost = remaining
        return ControlAction(
            proposed_intra,
            proposed_inter,
            max(0.0, raw_cost),
            "rule update from diffusion, coverage, reports, and benign loss",
        )
