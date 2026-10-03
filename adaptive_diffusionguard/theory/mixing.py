"""Community mixing measurements and the directed-OASIS adaptation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

DirectedAdaptation = Literal["symmetrize", "arcs"]


@dataclass(frozen=True, slots=True)
class CommunityConnectionStatistics:
    community: str
    node_count: int
    intra_incidents: int
    inter_incidents: int
    isolated_nodes: int


@dataclass(frozen=True, slots=True)
class CommunityMixingStatistics:
    intra_edge_count: int
    inter_edge_count: int
    total_edge_count: int
    mu: float | None
    directed_input: bool
    directed_adaptation: DirectedAdaptation
    self_loops_ignored: int
    duplicate_edges_ignored: int
    isolated_node_ids: tuple[int, ...]
    per_community: tuple[CommunityConnectionStatistics, ...]
    provenance: str = "project_adaptation_of_paper_eq_1"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_mixing_statistics(
    edges: Iterable[tuple[int, int]],
    communities: Mapping[int, str],
    *,
    directed_input: bool = True,
    directed_adaptation: DirectedAdaptation = "symmetrize",
) -> CommunityMixingStatistics:
    """Measure paper Eq. (1) after an explicit directed-network adaptation.

    ``symmetrize`` is the default for OASIS follower arcs: reciprocal arcs and
    duplicate arcs collapse to one unordered edge. ``arcs`` counts every
    distinct directed arc. Self-loops are always excluded because the paper's
    contact-network construction excludes them. An edgeless graph has
    undefined mixing, represented by ``mu=None`` rather than an invented zero.
    """

    if directed_adaptation not in {"symmetrize", "arcs"}:
        raise ValueError("directed_adaptation must be 'symmetrize' or 'arcs'")
    normalized_communities = {int(k): str(v) for k, v in communities.items()}

    seen: set[tuple[int, int]] = set()
    retained: list[tuple[int, int]] = []
    self_loops = 0
    duplicates = 0
    for raw_source, raw_target in edges:
        source, target = int(raw_source), int(raw_target)
        if source not in normalized_communities or target not in normalized_communities:
            raise ValueError(f"missing community label for edge ({source}, {target})")
        if source == target:
            self_loops += 1
            continue
        if directed_input and directed_adaptation == "arcs":
            key = (source, target)
        else:
            key = (min(source, target), max(source, target))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        retained.append((source, target))

    intra = 0
    inter = 0
    incidents: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    incident_nodes: set[int] = set()
    for source, target in retained:
        source_community = normalized_communities[source]
        target_community = normalized_communities[target]
        incident_nodes.update((source, target))
        if source_community == target_community:
            intra += 1
            incidents[source_community][0] += 2
        else:
            inter += 1
            incidents[source_community][1] += 1
            incidents[target_community][1] += 1

    isolated = tuple(sorted(set(normalized_communities) - incident_nodes))
    nodes_by_community: dict[str, list[int]] = defaultdict(list)
    for node_id, community in normalized_communities.items():
        nodes_by_community[community].append(node_id)
    isolated_set = set(isolated)
    per_community = tuple(
        CommunityConnectionStatistics(
            community=community,
            node_count=len(node_ids),
            intra_incidents=incidents[community][0],
            inter_incidents=incidents[community][1],
            isolated_nodes=sum(node_id in isolated_set for node_id in node_ids),
        )
        for community, node_ids in sorted(nodes_by_community.items())
    )
    total = intra + inter
    return CommunityMixingStatistics(
        intra_edge_count=intra,
        inter_edge_count=inter,
        total_edge_count=total,
        mu=None if total == 0 else inter / total,
        directed_input=directed_input,
        directed_adaptation=directed_adaptation,
        self_loops_ignored=self_loops,
        duplicate_edges_ignored=duplicates,
        isolated_node_ids=isolated,
        per_community=per_community,
    )
