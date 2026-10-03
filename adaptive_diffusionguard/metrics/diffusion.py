"""Windowed diffusion metrics and cascade-level extension points."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

from .state import GovernanceState


def state_from_database(
    connection: sqlite3.Connection,
    run_id: str,
    timestep: int,
    window: int,
    total_communities: int,
    cumulative_cost: float,
    total_budget: float,
) -> GovernanceState:
    """Compute controller inputs from actual impressions and OASIS actions."""
    start = max(0, timestep - window + 1)
    rows = connection.execute(
        """
        SELECT user_community, author_community, risk_score, shown
        FROM diffusionguard_impression
        WHERE run_id = ? AND timestep BETWEEN ? AND ?
        """,
        (run_id, start, timestep),
    ).fetchall()
    shown = [row for row in rows if row[3]]
    risky_shown = [row for row in shown if row[2] > 0]
    risk_adoption = len(risky_shown) / len(shown) if shown else 0.0
    cross_ratio = (
        sum(row[0] != row[1] for row in shown) / len(shown) if shown else 0.0
    )
    covered = {row[0] for row in risky_shown}
    coverage = len(covered) / max(1, total_communities)
    benign = [row for row in rows if row[2] == 0]
    benign_loss = (
        sum(not bool(row[3]) for row in benign) / len(benign) if benign else 0.0
    )

    trace_rows = connection.execute(
        "SELECT action FROM trace WHERE CAST(created_at AS INTEGER) BETWEEN ? AND ?",
        (start, timestep),
    ).fetchall()
    reposts = sum(row[0] in {"repost", "quote_post"} for row in trace_rows)
    reports = sum(row[0] == "report_post" for row in trace_rows)
    report_rate = reports / len(trace_rows) if trace_rows else 0.0
    return GovernanceState(
        risk_adoption_rate=min(1.0, risk_adoption),
        recent_reposts=reposts,
        community_coverage_ratio=min(1.0, coverage),
        cross_community_exposure_ratio=min(1.0, cross_ratio),
        report_rate=min(1.0, report_rate),
        benign_exposure_loss=min(1.0, benign_loss),
        cumulative_intervention_cost=cumulative_cost,
        remaining_budget=max(0.0, total_budget - cumulative_cost),
    )


class CascadeMetric(ABC):
    """Interface for later cascade-level evaluation; no results are fabricated."""

    @abstractmethod
    def compute(self, connection: sqlite3.Connection, root_post_id: int) -> float:
        """Compute a cascade statistic from persisted simulation data."""


class CascadeSize(CascadeMetric):
    def compute(self, connection: sqlite3.Connection, root_post_id: int) -> float:
        row = connection.execute(
            "SELECT COUNT(*) FROM post WHERE post_id = ? OR original_post_id = ?",
            (root_post_id, root_post_id),
        ).fetchone()
        return float(row[0] if row else 0)
