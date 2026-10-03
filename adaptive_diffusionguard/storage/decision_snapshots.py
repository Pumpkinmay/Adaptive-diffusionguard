"""Project-owned persistence for immutable model decision contexts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from adaptive_diffusionguard.llm.structured_actions import DecisionSnapshot

SECRET_PATTERN = re.compile(r"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")
SAFE_FAILURE_CATEGORY = re.compile(r"^[a-z0-9_]{1,64}$")
VALID_STATUSES = frozenset({"pending", "succeeded", "failed"})


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_secret_patterns(*values: str | None) -> None:
    if any(value and SECRET_PATTERN.search(value) for value in values):
        raise ValueError("decision snapshot contains a credential-like pattern")


@dataclass(frozen=True, slots=True)
class PendingDecision:
    decision_id: str
    run_id: str
    user_id: int
    agent_id: int
    timestep: int
    decision_sequence: int


class DecisionSnapshotStore:
    """Persist one row per logical decision; provider retries reuse the same row."""

    TABLE = "diffusionguard_decision_snapshot"

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        self.connection = connection
        self._now = now_factory or _utc_now
        self.initialize()

    def initialize(self) -> None:
        """Apply an idempotent project-owned SQLite migration."""
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                timestep INTEGER NOT NULL,
                decision_sequence INTEGER NOT NULL,
                feed_json TEXT NOT NULL,
                visible_post_ids_json TEXT NOT NULL,
                legal_choice_ids_json TEXT NOT NULL,
                state_identifier TEXT NOT NULL,
                response_format_hash TEXT NOT NULL,
                prompt_context_hash TEXT NOT NULL,
                user_profile TEXT NOT NULL,
                community TEXT NOT NULL,
                behavior_history_json TEXT NOT NULL,
                neighbor_interactions_json TEXT NOT NULL,
                platform_notice TEXT NOT NULL,
                rationale TEXT,
                selected_choice_id TEXT,
                action_trace_rowid INTEGER UNIQUE,
                status TEXT NOT NULL
                    CHECK(status IN ('pending', 'succeeded', 'failed')),
                failure_category TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE(run_id, user_id, timestep, decision_sequence)
            )
            """
        )
        self.connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_status_sequence "
            f"ON {self.TABLE}(status, decision_sequence, decision_id)"
        )
        self.connection.commit()

    @staticmethod
    def decision_id(
        *, run_id: str, user_id: int, timestep: int, decision_sequence: int
    ) -> str:
        return (
            f"{run_id}:t{int(timestep):06d}:u{int(user_id):06d}:"
            f"d{int(decision_sequence):06d}"
        )

    def create_pending(
        self,
        snapshot: DecisionSnapshot,
        *,
        run_id: str,
        agent_id: int,
        decision_sequence: int,
        user_profile: str,
        community: str,
        behavior_history: list[str],
        neighbor_interactions: list[str] | None = None,
        platform_notice: str = "",
    ) -> PendingDecision:
        """Write the frozen context before any provider request is issued."""
        decision_id = self.decision_id(
            run_id=run_id,
            user_id=snapshot.user_id,
            timestep=snapshot.timestep,
            decision_sequence=decision_sequence,
        )
        feed_json = snapshot.feed.decode("utf-8")
        visible_json = canonical_json(list(snapshot.visible_post_ids))
        choices_json = canonical_json(list(snapshot.legal_choice_ids))
        history_json = canonical_json(list(behavior_history))
        neighbors_json = canonical_json(list(neighbor_interactions or []))
        _reject_secret_patterns(
            decision_id,
            run_id,
            feed_json,
            visible_json,
            choices_json,
            snapshot.state_identifier,
            user_profile,
            community,
            history_json,
            neighbors_json,
            platform_notice,
        )
        with self.connection:
            self.connection.execute(
                f"""
                INSERT INTO {self.TABLE} (
                    decision_id, run_id, user_id, agent_id, timestep,
                    decision_sequence, feed_json, visible_post_ids_json,
                    legal_choice_ids_json, state_identifier,
                    response_format_hash, prompt_context_hash, user_profile,
                    community, behavior_history_json,
                    neighbor_interactions_json, platform_notice, rationale,
                    selected_choice_id, action_trace_rowid, status,
                    failure_category, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          NULL, NULL, NULL, 'pending', NULL, ?, NULL)
                """,
                (
                    decision_id,
                    run_id,
                    snapshot.user_id,
                    int(agent_id),
                    snapshot.timestep,
                    int(decision_sequence),
                    feed_json,
                    visible_json,
                    choices_json,
                    snapshot.state_identifier,
                    _sha256(snapshot.response_format),
                    _sha256(snapshot.messages),
                    user_profile,
                    community,
                    history_json,
                    neighbors_json,
                    platform_notice,
                    self._now(),
                ),
            )
        return PendingDecision(
            decision_id=decision_id,
            run_id=run_id,
            user_id=snapshot.user_id,
            agent_id=int(agent_id),
            timestep=snapshot.timestep,
            decision_sequence=int(decision_sequence),
        )

    def mark_failed(self, decision_id: str, failure_category: str) -> None:
        """Retain a failed logical decision without storing provider error text."""
        if not SAFE_FAILURE_CATEGORY.fullmatch(failure_category):
            raise ValueError(
                "failure_category must be a non-sensitive stable identifier"
            )
        with self.connection:
            cursor = self.connection.execute(
                f"""
                UPDATE {self.TABLE}
                SET status = 'failed', failure_category = ?, completed_at = ?
                WHERE decision_id = ? AND status = 'pending'
                """,
                (failure_category, self._now(), decision_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "decision snapshot is missing or is no longer pending"
                )

    def mark_succeeded(
        self,
        decision_id: str,
        *,
        selected_choice_id: str,
        rationale: str,
        action_trace_rowid: int,
    ) -> None:
        """Atomically bind a confirmed OASIS trace to the pending decision."""
        _reject_secret_patterns(selected_choice_id, rationale)
        with self.connection:
            row = self.connection.execute(
                f"""
                SELECT user_id, legal_choice_ids_json
                FROM {self.TABLE}
                WHERE decision_id = ? AND status = 'pending'
                """,
                (decision_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "decision snapshot is missing or is no longer pending"
                )
            user_id = int(row[0])
            legal_choices = json.loads(row[1])
            if selected_choice_id not in legal_choices:
                raise ValueError(
                    "selected choice is absent from persisted legal choices"
                )
            action_name = selected_choice_id.partition(":")[0]
            expected_action = {
                "repost": "repost",
                "quote": "quote_post",
                "report": "report_post",
                "ignore": "do_nothing",
            }.get(action_name)
            if expected_action is None:
                raise ValueError("selected choice has an unsupported action")
            trace = self.connection.execute(
                "SELECT user_id, action FROM trace WHERE rowid = ?",
                (int(action_trace_rowid),),
            ).fetchone()
            if trace is None or int(trace[0]) != user_id or trace[1] != expected_action:
                raise ValueError("action trace does not match the persisted decision")
            cursor = self.connection.execute(
                f"""
                UPDATE {self.TABLE}
                SET rationale = ?, selected_choice_id = ?,
                    action_trace_rowid = ?, status = 'succeeded',
                    failure_category = NULL, completed_at = ?
                WHERE decision_id = ? AND status = 'pending'
                """,
                (
                    rationale,
                    selected_choice_id,
                    int(action_trace_rowid),
                    self._now(),
                    decision_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("failed to complete decision snapshot atomically")

    def status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            f"SELECT status, COUNT(*) FROM {self.TABLE} GROUP BY status"
        ).fetchall()
        counts = {status: 0 for status in sorted(VALID_STATUSES)}
        counts.update({str(status): int(count) for status, count in rows})
        return counts


__all__ = [
    "VALID_STATUSES",
    "DecisionSnapshotStore",
    "PendingDecision",
    "canonical_json",
]
