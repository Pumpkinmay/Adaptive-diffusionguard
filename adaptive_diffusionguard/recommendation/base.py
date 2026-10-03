"""Recommendation candidate abstractions."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    """One content item before exposure governance is applied."""

    post_id: int
    base_score: float
    source: str


class CandidateGenerator(ABC):
    """Generate the exact candidate set presented to a policy."""

    @abstractmethod
    def generate(
        self,
        connection: sqlite3.Connection,
        user_id: int,
        recommendation_count: int,
        following_count: int,
    ) -> list[Candidate]:
        """Return recommendation and following-feed candidates."""
