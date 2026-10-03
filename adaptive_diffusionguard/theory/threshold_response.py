"""Deterministic paper-threshold response and the OASIS exposure bridge.

The paper model is an undirected, irreversible binary threshold process.  The
main project experiment keeps paper influence weights fixed and lets OASIS
retention control only which adopted neighbours are observable.  This is a
project adaptation, not an identity between paper omega and feed retention.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from .allocation import paper_exponential_cost, project_l1_cost

ControlDirection = Literal["intra", "balanced", "inter"]


def _unit(name: str, value: float) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def symmetrize_contacts(
    edges: Iterable[tuple[int, int]], users: Iterable[int]
) -> dict[int, frozenset[int]]:
    """Adapt directed OASIS arcs to unique undirected paper contacts."""

    user_ids = {int(user) for user in users}
    neighbours: dict[int, set[int]] = {user: set() for user in user_ids}
    for raw_source, raw_target in edges:
        source, target = int(raw_source), int(raw_target)
        if source not in user_ids or target not in user_ids:
            raise ValueError(f"edge ({source}, {target}) references an unknown user")
        if source == target:
            continue
        neighbours[source].add(target)
        neighbours[target].add(source)
    return {user: frozenset(values) for user, values in neighbours.items()}


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    timestep: int
    adopted_by_root: Mapping[int, frozenset[int]]
    provenance: str = "paper_synchronous_frozen_adoption_state"


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    timestep: int
    user_id: int
    root_post_id: int
    user_community: str
    root_author_community: str
    intra_adopted_neighbor_count: int
    inter_adopted_neighbor_count: int
    observable_intra_adopted_neighbor_count: int
    observable_inter_adopted_neighbor_count: int
    total_degree: int
    intra_degree: int
    inter_degree: int
    threshold: float
    threshold_boundary: float
    paper_omega_intra: float
    paper_omega_inter: float
    paper_native_signal: float
    observable_signal: float
    paper_native_threshold_satisfied: bool
    observable_threshold_satisfied: bool
    threshold_satisfied: bool
    exposure_gate_enabled: bool
    exposure_gate_passed: bool
    threshold_met_but_exposure_blocked: bool
    already_adopted: bool
    should_attempt_adoption: bool
    final_adopted: bool = False
    dispatcher_success: bool = False
    provenance: str = "project_adaptation_observable_neighbour_threshold"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrictAllocation:
    mu: float
    direction: ControlDirection
    tolerance: float
    minimum_strict_gap: float
    project_budget: float
    eligible_oasis_keep_pairs: tuple[tuple[float, float], ...]
    provenance: str = "paper_direction_project_strict_keep_allocation"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def allocate_strict_oasis_keep(
    *,
    mu: float,
    project_budget: float,
    keep_grid: Iterable[tuple[float, float]],
    tolerance: float,
    minimum_strict_gap: float,
) -> StrictAllocation:
    """Apply a strict paper-guided direction to project keep candidates."""

    mixing = _unit("mu", mu)
    budget = float(project_budget)
    gap = float(minimum_strict_gap)
    if not 0.0 <= budget <= 2.0:
        raise ValueError("project_budget must be in [0, 2]")
    if not 0.0 <= tolerance < 0.5:
        raise ValueError("tolerance must be in [0, 0.5)")
    if not 0.0 < gap <= 1.0:
        raise ValueError("minimum_strict_gap must be in (0, 1]")
    if mixing < 0.5 - tolerance:
        direction: ControlDirection = "intra"
    elif mixing > 0.5 + tolerance:
        direction = "inter"
    else:
        direction = "balanced"

    eligible: set[tuple[float, float]] = set()
    for raw_intra, raw_inter in keep_grid:
        intra = _unit("oasis_keep_intra", raw_intra)
        inter = _unit("oasis_keep_inter", raw_inter)
        if project_l1_cost(intra, inter) > budget + 1e-12:
            continue
        if direction == "intra":
            consistent = inter - intra >= gap - 1e-12
        elif direction == "inter":
            consistent = intra - inter >= gap - 1e-12
        else:
            consistent = math.isclose(intra, inter, abs_tol=1e-12)
        if consistent:
            eligible.add((intra, inter))
    if not eligible:
        raise ValueError(
            f"no keep candidate satisfies strict {direction} direction and budget"
        )
    return StrictAllocation(
        mu=mixing,
        direction=direction,
        tolerance=tolerance,
        minimum_strict_gap=gap,
        project_budget=budget,
        eligible_oasis_keep_pairs=tuple(sorted(eligible)),
    )


class ThresholdResponseEngine:
    """Synchronous irreversible adoption state keyed by ``(user, root)``."""

    def __init__(
        self,
        *,
        communities: Mapping[int, str],
        contact_edges: Iterable[tuple[int, int]],
        initial_adopters: Mapping[int, Iterable[int]],
        threshold: float,
        paper_omega_intra: float,
        paper_omega_inter: float,
        exposure_gate_enabled: bool,
    ) -> None:
        self.communities = {int(user): str(value) for user, value in communities.items()}
        self.neighbours = symmetrize_contacts(contact_edges, self.communities)
        self.threshold = _unit("threshold", threshold)
        self.paper_omega_intra = _unit("paper_omega_intra", paper_omega_intra)
        self.paper_omega_inter = _unit("paper_omega_inter", paper_omega_inter)
        self.exposure_gate_enabled = bool(exposure_gate_enabled)
        self._adopted: dict[int, set[int]] = {}
        self._adoption_time: dict[tuple[int, int], int] = {}
        for raw_root, raw_users in initial_adopters.items():
            root = int(raw_root)
            users = {int(user) for user in raw_users}
            unknown = users - set(self.communities)
            if unknown:
                raise ValueError(f"initial adopters contain unknown users: {unknown}")
            self._adopted[root] = users
            for user in users:
                self._adoption_time[user, root] = 0
        self._snapshot: DecisionSnapshot | None = None

    def begin_timestep(self, timestep: int) -> DecisionSnapshot:
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        self._snapshot = DecisionSnapshot(
            timestep=int(timestep),
            adopted_by_root={
                root: frozenset(users) for root, users in sorted(self._adopted.items())
            },
        )
        return self._snapshot

    def evaluate(
        self,
        *,
        user_id: int,
        root_post_id: int,
        root_author_community: str,
        observable_adopter_ids: Iterable[int],
    ) -> ThresholdEvaluation:
        if self._snapshot is None:
            raise RuntimeError("begin_timestep must be called before evaluate")
        user = int(user_id)
        root = int(root_post_id)
        if user not in self.communities:
            raise ValueError(f"unknown user {user}")
        frozen_adopters = self._snapshot.adopted_by_root.get(root, frozenset())
        neighbours = self.neighbours[user]
        adopted_neighbours = neighbours & frozen_adopters
        observable = {
            int(neighbour) for neighbour in observable_adopter_ids
        } & adopted_neighbours
        user_community = self.communities[user]
        intra_degree = sum(
            self.communities[neighbour] == user_community for neighbour in neighbours
        )
        inter_degree = len(neighbours) - intra_degree

        def split(values: Iterable[int]) -> tuple[int, int]:
            intra = sum(self.communities[value] == user_community for value in values)
            return intra, len(set(values)) - intra

        intra, inter = split(adopted_neighbours)
        observable_intra, observable_inter = split(observable)
        native_signal = (
            self.paper_omega_intra * intra + self.paper_omega_inter * inter
        )
        observable_signal = (
            self.paper_omega_intra * observable_intra
            + self.paper_omega_inter * observable_inter
        )
        boundary = self.threshold * len(neighbours)
        native_satisfied = native_signal > boundary
        observable_satisfied = observable_signal > boundary
        selected_satisfied = (
            observable_satisfied if self.exposure_gate_enabled else native_satisfied
        )
        already_adopted = user in frozen_adopters
        gate_passed = not self.exposure_gate_enabled or bool(observable)
        return ThresholdEvaluation(
            timestep=self._snapshot.timestep,
            user_id=user,
            root_post_id=root,
            user_community=user_community,
            root_author_community=str(root_author_community),
            intra_adopted_neighbor_count=intra,
            inter_adopted_neighbor_count=inter,
            observable_intra_adopted_neighbor_count=observable_intra,
            observable_inter_adopted_neighbor_count=observable_inter,
            total_degree=len(neighbours),
            intra_degree=intra_degree,
            inter_degree=inter_degree,
            threshold=self.threshold,
            threshold_boundary=boundary,
            paper_omega_intra=self.paper_omega_intra,
            paper_omega_inter=self.paper_omega_inter,
            paper_native_signal=native_signal,
            observable_signal=observable_signal,
            paper_native_threshold_satisfied=native_satisfied,
            observable_threshold_satisfied=observable_satisfied,
            threshold_satisfied=selected_satisfied,
            exposure_gate_enabled=self.exposure_gate_enabled,
            exposure_gate_passed=gate_passed,
            threshold_met_but_exposure_blocked=(
                self.exposure_gate_enabled
                and native_satisfied
                and not observable_satisfied
            ),
            already_adopted=already_adopted,
            should_attempt_adoption=selected_satisfied and not already_adopted,
        )

    def commit(
        self, timestep: int, successful_adoptions: Iterable[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]:
        if self._snapshot is None or self._snapshot.timestep != timestep:
            raise RuntimeError("commit must match the frozen timestep")
        unique = sorted({(int(user), int(root)) for user, root in successful_adoptions})
        committed: list[tuple[int, int]] = []
        for user, root in unique:
            if user in self._snapshot.adopted_by_root.get(root, frozenset()):
                continue
            self._adopted.setdefault(root, set()).add(user)
            self._adoption_time[user, root] = timestep
            committed.append((user, root))
        self._snapshot = None
        return tuple(committed)

    def adopters(self, root_post_id: int) -> frozenset[int]:
        return frozenset(self._adopted.get(int(root_post_id), set()))

    def adoption_time(self, user_id: int, root_post_id: int) -> int | None:
        return self._adoption_time.get((int(user_id), int(root_post_id)))


def simulate_paper_native(
    *,
    communities: Mapping[int, str],
    contact_edges: Iterable[tuple[int, int]],
    initial_adopters: Iterable[int],
    threshold: float,
    paper_omega_intra: float,
    paper_omega_inter: float,
    timesteps: int,
) -> dict[str, object]:
    """Run the synchronous paper-native process with exposure gating disabled."""

    engine = ThresholdResponseEngine(
        communities=communities,
        contact_edges=contact_edges,
        initial_adopters={0: initial_adopters},
        threshold=threshold,
        paper_omega_intra=paper_omega_intra,
        paper_omega_inter=paper_omega_inter,
        exposure_gate_enabled=False,
    )
    by_timestep = []
    for timestep in range(1, timesteps + 1):
        snapshot = engine.begin_timestep(timestep)
        frozen = snapshot.adopted_by_root[0]
        eligible = []
        for user in sorted(communities):
            evaluation = engine.evaluate(
                user_id=user,
                root_post_id=0,
                root_author_community="community-a",
                observable_adopter_ids=(),
            )
            if evaluation.should_attempt_adoption:
                eligible.append((user, 0))
        committed = engine.commit(timestep, eligible)
        by_timestep.append(
            {
                "timestep": timestep,
                "frozen_adopters": len(frozen),
                "new_adopters": len(committed),
                "total_adopters": len(engine.adopters(0)),
            }
        )
        if not committed:
            break
    return {
        "final_adopters": sorted(engine.adopters(0)),
        "final_adoption_fraction": len(engine.adopters(0)) / len(communities),
        "timesteps": by_timestep,
        "threshold": threshold,
        "paper_omega_intra": paper_omega_intra,
        "paper_omega_inter": paper_omega_inter,
        "paper_cost": paper_exponential_cost(
            paper_omega_intra, paper_omega_inter
        ),
        "exposure_gate_enabled": False,
        "provenance": "paper_native_synchronous_threshold_check",
    }
