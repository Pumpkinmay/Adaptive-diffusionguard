"""Normalized SQLite impression log."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ImpressionRecord:
    run_id: str
    timestep: int
    user_id: int
    post_id: int
    root_post_id: int
    user_community: str
    author_community: str
    base_score: float
    risk_score: float
    omega_intra: float
    omega_inter: float
    keep_probability: float
    shown: bool
    source: str = "recommendation"


class ImpressionStore:
    COLUMNS = tuple(ImpressionRecord.__dataclass_fields__)

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diffusionguard_impression (
                impression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestep INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                post_id INTEGER NOT NULL,
                root_post_id INTEGER NOT NULL,
                user_community TEXT NOT NULL,
                author_community TEXT NOT NULL,
                base_score REAL NOT NULL,
                risk_score REAL NOT NULL CHECK(risk_score BETWEEN 0 AND 1),
                omega_intra REAL NOT NULL CHECK(omega_intra BETWEEN 0 AND 1),
                omega_inter REAL NOT NULL CHECK(omega_inter BETWEEN 0 AND 1),
                keep_probability REAL NOT NULL
                    CHECK(keep_probability BETWEEN 0 AND 1),
                shown INTEGER NOT NULL CHECK(shown IN (0, 1)),
                source TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def append(self, record: ImpressionRecord) -> None:
        values = asdict(record)
        placeholders = ", ".join("?" for _ in self.COLUMNS)
        self.connection.execute(
            f"INSERT INTO diffusionguard_impression ({', '.join(self.COLUMNS)}) "
            f"VALUES ({placeholders})",
            tuple(values[column] for column in self.COLUMNS),
        )
        self.connection.commit()

    def append_many(self, records: Iterable[ImpressionRecord]) -> None:
        rows = [tuple(asdict(r)[column] for column in self.COLUMNS) for r in records]
        if not rows:
            return
        placeholders = ", ".join("?" for _ in self.COLUMNS)
        self.connection.executemany(
            f"INSERT INTO diffusionguard_impression ({', '.join(self.COLUMNS)}) "
            f"VALUES ({placeholders})",
            rows,
        )
        self.connection.commit()
