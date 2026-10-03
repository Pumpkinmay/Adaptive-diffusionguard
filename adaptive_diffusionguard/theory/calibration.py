"""Offline OASIS exposure calibration for a theory-guided candidate grid."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .allocation import AllocationResult


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    omega_intra: float
    omega_inter: float
    seed: int
    high_risk_exposures: int
    intra_high_risk_exposures: int
    inter_high_risk_exposures: int
    successful_risk_reposts: int
    cascade_size: int
    community_coverage: float
    benign_exposure_loss: float
    intervention_cost: float
    provenance: str = "project_adaptation_oasis_observation"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    omega_intra: float
    omega_inter: float
    observations_used: int
    mean_high_risk_exposures: float
    mean_successful_risk_reposts: float
    mean_benign_exposure_loss: float
    direction: str
    reason: str
    provenance: str = "project_adaptation_not_paper_equivalence"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def select_calibrated_keep_probabilities(
    observations: list[CalibrationObservation],
    allocation: AllocationResult,
) -> CalibrationSelection:
    """Select the best observed direction-consistent candidate deterministically."""

    allowed = {
        (candidate.omega_intra, candidate.omega_inter)
        for candidate in allocation.candidates
        if candidate.direction_consistent
    }
    grouped: dict[tuple[float, float], list[CalibrationObservation]] = defaultdict(list)
    for observation in observations:
        key = (observation.omega_intra, observation.omega_inter)
        if observation.provenance != "project_adaptation_oasis_observation":
            raise ValueError("calibration observations must be marked project adaptation")
        if key in allowed:
            grouped[key].append(observation)
    if not grouped:
        raise ValueError("no direction-consistent calibration observation is available")

    ranked: list[tuple[tuple[float, ...], tuple[float, float], tuple[float, ...]]] = []
    for key, rows in grouped.items():
        count = len(rows)
        means = (
            sum(row.high_risk_exposures for row in rows) / count,
            sum(row.successful_risk_reposts for row in rows) / count,
            sum(row.community_coverage for row in rows) / count,
            sum(row.benign_exposure_loss for row in rows) / count,
            sum(row.intervention_cost for row in rows) / count,
        )
        ranked.append(((means[0], means[1], means[2], means[3], means[4], *key), key, means))
    _, selected_key, selected_means = min(ranked, key=lambda item: item[0])
    selected_rows = grouped[selected_key]
    return CalibrationSelection(
        omega_intra=selected_key[0],
        omega_inter=selected_key[1],
        observations_used=len(selected_rows),
        mean_high_risk_exposures=selected_means[0],
        mean_successful_risk_reposts=selected_means[1],
        mean_benign_exposure_loss=selected_means[3],
        direction=allocation.direction,
        reason=(
            "minimum observed high-risk exposure, then risk reposts, coverage, "
            "benign loss, and intervention cost within the shared budget"
        ),
    )
