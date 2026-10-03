"""OASIS SQLite-backed candidate generation."""

from __future__ import annotations

import random
import sqlite3

from .base import Candidate, CandidateGenerator


class OASISCandidateGenerator(CandidateGenerator):
    """Reproduce OASIS refresh selection with an isolated seeded RNG.

    Recommendation-table entries are sampled, while followed-user posts are
    ranked by likes. Both sources are retained so the governance policy sees
    everything that could actually enter the feed.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        connection: sqlite3.Connection,
        user_id: int,
        recommendation_count: int,
        following_count: int,
    ) -> list[Candidate]:
        cursor = connection.cursor()
        rec_ids = [
            int(row[0])
            for row in cursor.execute(
                "SELECT post_id FROM rec WHERE user_id = ? ORDER BY post_id",
                (user_id,),
            ).fetchall()
        ]
        if len(rec_ids) > recommendation_count:
            rec_ids = self._rng.sample(rec_ids, recommendation_count)

        following_rows = cursor.execute(
            """
            SELECT post.post_id, post.num_likes
            FROM post
            JOIN follow ON post.user_id = follow.followee_id
            WHERE follow.follower_id = ?
            ORDER BY post.num_likes DESC, post.post_id DESC
            LIMIT ?
            """,
            (user_id, following_count),
        ).fetchall()

        candidates: list[Candidate] = []
        seen: set[int] = set()
        for rank, post_id in enumerate(rec_ids):
            if post_id not in seen:
                candidates.append(
                    Candidate(int(post_id), 1.0 / (rank + 1), "recommendation")
                )
                seen.add(post_id)
        for rank, (post_id, likes) in enumerate(following_rows):
            post_id = int(post_id)
            if post_id not in seen:
                # A monotone and documented proxy; OASIS does not persist its
                # raw recommendation scores in the rec table.
                score = float(likes) + 1.0 / (rank + 1)
                candidates.append(Candidate(post_id, score, "following"))
                seen.add(post_id)
        return candidates
