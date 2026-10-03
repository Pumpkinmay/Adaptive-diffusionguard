"""Theory bridge utilities with explicit paper/adaptation provenance."""

from .allocation import (
    AllocationResult,
    ControlCandidate,
    allocate_cosref_control,
    paper_exponential_cost,
    project_l1_cost,
)
from .calibration import (
    CalibrationObservation,
    CalibrationSelection,
    select_calibrated_keep_probabilities,
)
from .mixing import CommunityMixingStatistics, compute_mixing_statistics

__all__ = [
    "AllocationResult",
    "CalibrationObservation",
    "CalibrationSelection",
    "CommunityMixingStatistics",
    "ControlCandidate",
    "allocate_cosref_control",
    "compute_mixing_statistics",
    "paper_exponential_cost",
    "project_l1_cost",
    "select_calibrated_keep_probabilities",
]
