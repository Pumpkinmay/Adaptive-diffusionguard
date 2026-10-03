"""Theory-guided allocation without equating paper omega to feed retention."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

ControlDirection = Literal["intra", "balanced", "inter"]


def _unit(name: str, value: float) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def paper_exponential_cost(omega_intra: float, omega_inter: float) -> float:
    """Paper Eq. (5); zero means no regulation and one maximal regulation."""

    intra = _unit("omega_intra", omega_intra)
    inter = _unit("omega_inter", omega_inter)
    return (math.exp(-(intra + inter)) - math.exp(-2.0)) / (
        1.0 - math.exp(-2.0)
    )


def project_l1_cost(omega_intra: float, omega_inter: float) -> float:
    """Shared project budget, not a paper equation."""

    intra = _unit("omega_intra", omega_intra)
    inter = _unit("omega_inter", omega_inter)
    return (1.0 - intra) + (1.0 - inter)


@dataclass(frozen=True, slots=True)
class ControlCandidate:
    omega_intra: float
    omega_inter: float
    project_budget_cost: float
    paper_cost: float
    direction_consistent: bool
    provenance: str = "project_adaptation_candidate"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AllocationResult:
    mu: float
    tolerance: float
    direction: ControlDirection
    project_budget: float
    candidates: tuple[ControlCandidate, ...]
    reason: str
    paper_reference: str
    paper_reference_omega: tuple[float, float] | None
    provenance: str = "paper_direction_plus_project_budget_adaptation"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_PAPER_REFERENCE_POINTS = {
    0.2: (0.00, 0.85),
    0.4: (0.23, 0.34),
    0.6: (0.38, 0.21),
    0.8: (0.78, 0.10),
}


def allocate_cosref_control(
    mu: float,
    project_budget: float,
    omega_grid: list[tuple[float, float]],
    *,
    tolerance: float = 0.05,
) -> AllocationResult:
    """Filter an OASIS scan grid by budget and paper-supported direction.

    This function does not claim that candidate keep probabilities are paper
    transmissibilities. It carries only the directional prior from Fig. 3 and
    Supplementary Fig. 16 into a project-side calibration search.
    """

    mixing = _unit("mu", mu)
    budget = float(project_budget)
    if not 0.0 <= budget <= 2.0:
        raise ValueError("project_budget must be in [0, 2]")
    if not 0.0 <= tolerance < 0.5:
        raise ValueError("tolerance must be in [0, 0.5)")
    if mixing < 0.5 - tolerance:
        direction: ControlDirection = "intra"
        reason = "mu is below the transition band; prioritize stronger intra control"
    elif mixing > 0.5 + tolerance:
        direction = "inter"
        reason = "mu is above the transition band; prioritize stronger inter control"
    else:
        direction = "balanced"
        reason = "mu lies in the declared transition band; do not impose one-sided control"

    candidates: list[ControlCandidate] = []
    seen: set[tuple[float, float]] = set()
    for raw_intra, raw_inter in omega_grid:
        intra = _unit("omega_intra", raw_intra)
        inter = _unit("omega_inter", raw_inter)
        key = (intra, inter)
        if key in seen:
            continue
        seen.add(key)
        l1_cost = project_l1_cost(intra, inter)
        if l1_cost > budget + 1e-12:
            continue
        consistent = (
            (direction == "intra" and intra <= inter)
            or (direction == "inter" and inter <= intra)
            or direction == "balanced"
        )
        candidates.append(
            ControlCandidate(
                omega_intra=intra,
                omega_inter=inter,
                project_budget_cost=l1_cost,
                paper_cost=paper_exponential_cost(intra, inter),
                direction_consistent=consistent,
            )
        )
    if not candidates:
        raise ValueError("omega_grid has no candidate inside the project budget")
    candidates.sort(
        key=lambda item: (
            not item.direction_consistent,
            -item.project_budget_cost,
            item.omega_intra,
            item.omega_inter,
        )
    )
    reference_omega = next(
        (
            value
            for reference_mu, value in _PAPER_REFERENCE_POINTS.items()
            if math.isclose(mixing, reference_mu, abs_tol=1e-12)
        ),
        None,
    )
    return AllocationResult(
        mu=mixing,
        tolerance=tolerance,
        direction=direction,
        project_budget=budget,
        candidates=tuple(candidates),
        reason=reason,
        paper_reference="Nature Fig. 3 and Supplementary Fig. 16",
        paper_reference_omega=reference_omega,
    )
