"""OASIS platform subclass with end-to-end exposure control."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from uuid import uuid4

from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, RecsysType

from adaptive_diffusionguard.governance.controller import Controller
from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy
from adaptive_diffusionguard.metrics.diffusion import state_from_database
from adaptive_diffusionguard.metrics.state import GovernanceState
from adaptive_diffusionguard.recommendation.base import CandidateGenerator
from adaptive_diffusionguard.recommendation.candidate_generator import (
    OASISCandidateGenerator,
)
from adaptive_diffusionguard.storage.impressions import (
    ImpressionRecord,
    ImpressionStore,
)

logger = logging.getLogger(__name__)


class AdaptiveDiffusionPlatform(Platform):
    """Apply governance to recommendation and followed-user content.

    User community and risk mappings are supplied by experiment data/config;
    they are never inferred from protected characteristics or hard-coded.
    """

    def __init__(
        self,
        db_path: str,
        *,
        user_communities: Mapping[int, str],
        post_risk_scores: Mapping[int, float],
        policy: StaticCOSREFPolicy,
        random_seed: int,
        controller: Controller | None = None,
        candidate_generator: CandidateGenerator | None = None,
        run_id: str | None = None,
        controller_window: int = 5,
        intervention_budget: float | None = None,
        state_provider: Callable[[int], GovernanceState] | None = None,
        **platform_kwargs: Any,
    ) -> None:
        super().__init__(db_path=db_path, **platform_kwargs)
        self.user_communities = {int(k): str(v) for k, v in user_communities.items()}
        self.post_risk_scores = {
            int(k): self._validate_risk(v) for k, v in post_risk_scores.items()
        }
        self.policy = policy
        self.controller = controller
        self.candidate_generator = candidate_generator or OASISCandidateGenerator(
            random_seed
        )
        self.run_id = run_id or uuid4().hex
        self.controller_window = max(1, controller_window)
        self.intervention_budget = (
            None if intervention_budget is None else max(0.0, intervention_budget)
        )
        self.state_provider = state_provider
        self.impressions = ImpressionStore(self.db)
        self._refresh_counter = 0
        self._last_control_timestep: int | None = None
        initial_cost = (1.0 - policy.omega_intra) + (1.0 - policy.omega_inter)
        if (
            self.intervention_budget is not None
            and initial_cost > self.intervention_budget + 1e-12
        ):
            raise ValueError(
                "initial policy exceeds intervention budget: "
                f"cost={initial_cost}, budget={self.intervention_budget}"
            )
        self._cumulative_intervention_cost = initial_cost

    @staticmethod
    def _validate_risk(value: float) -> float:
        risk = float(value)
        if not 0.0 <= risk <= 1.0:
            raise ValueError(f"risk score must be in [0, 1], got {risk}")
        return risk

    def _timestep(self) -> int:
        value = getattr(self.sandbox_clock, "time_step", None)
        if isinstance(value, int):
            return value
        self._refresh_counter += 1
        return self._refresh_counter

    def _current_time(self) -> Any:
        if self.recsys_type == RecsysType.REDDIT:
            # OASIS Clock and Platform use naive datetimes as their public
            # contract, so adding tzinfo here would make subtraction invalid.
            return self.sandbox_clock.time_transfer(
                datetime.now(),  # noqa: DTZ005
                self.start_time,
            )
        return self.sandbox_clock.get_time_step()

    def _format_feed_posts(self, rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Format feed rows while working around OASIS 0.2.5 quote handling."""
        posts: list[dict[str, Any]] = []
        for row in rows:
            (
                post_id,
                user_id,
                original_post_id,
                content,
                quote_content,
                created_at,
                num_likes,
                num_dislikes,
                num_shares,
            ) = row
            post_type = self.pl_utils._get_post_type(post_id)
            if post_type is None:
                continue

            feed_post_id = int(post_id)
            comments_post_id = int(post_id)
            report_row = self.db.execute(
                "SELECT num_reports FROM post WHERE post_id = ?", (post_id,)
            ).fetchone()
            num_reports = int(report_row[0]) if report_row is not None else 0

            if post_type["type"] == "repost":
                repost_id = feed_post_id
                original_user_row = self.db.execute(
                    "SELECT user_id FROM post WHERE post_id = ?",
                    (original_post_id,),
                ).fetchone()
                if original_user_row is None:
                    continue
                original_user_id = int(original_user_row[0])
                comments_post_id = int(post_type["root_post_id"])
                root_row = self.db.execute(
                    """
                    SELECT content, quote_content, created_at, num_likes,
                           num_dislikes, num_shares, num_reports
                    FROM post WHERE post_id = ?
                    """,
                    (comments_post_id,),
                ).fetchone()
                if root_row is None:
                    continue
                (
                    content,
                    quote_content,
                    created_at,
                    num_likes,
                    num_dislikes,
                    num_shares,
                    num_reports,
                ) = root_row
                feed_post_id = repost_id
                post_content = (
                    f"User {user_id} reposted a post from User "
                    f"{original_user_id}. Repost content: {content}. "
                )
            elif post_type["type"] == "quote":
                original_user_row = self.db.execute(
                    "SELECT user_id FROM post WHERE post_id = ?",
                    (original_post_id,),
                ).fetchone()
                if original_user_row is None:
                    continue
                original_user_id = int(original_user_row[0])
                post_content = (
                    f"User {user_id} quoted a post from User "
                    f"{original_user_id}. Quote content: {quote_content}. "
                    f"Original Content: {content}"
                )
            else:
                post_content = content

            comments_rows = self.db.execute(
                """
                SELECT comment_id, post_id, user_id, content, created_at,
                       num_likes, num_dislikes
                FROM comment WHERE post_id = ?
                """,
                (comments_post_id,),
            ).fetchall()
            comments = []
            for comment in comments_rows:
                (
                    comment_id,
                    comment_post_id,
                    comment_user_id,
                    comment_content,
                    comment_created_at,
                    comment_likes,
                    comment_dislikes,
                ) = comment
                formatted_comment = {
                    "comment_id": comment_id,
                    "post_id": comment_post_id,
                    "user_id": comment_user_id,
                    "content": comment_content,
                    "created_at": comment_created_at,
                }
                if self.show_score:
                    formatted_comment["score"] = comment_likes - comment_dislikes
                else:
                    formatted_comment["num_likes"] = comment_likes
                    formatted_comment["num_dislikes"] = comment_dislikes
                comments.append(formatted_comment)

            if num_reports >= self.report_threshold:
                post_content = (
                    f"[Warning: This post has been reported {num_reports} times]\n"
                    f"{post_content}"
                )

            formatted_post = {
                "post_id": feed_post_id,
                "user_id": user_id,
                "content": post_content,
                "created_at": created_at,
                "num_shares": num_shares,
                "num_reports": num_reports,
                "comments": comments,
            }
            if self.show_score:
                formatted_post["score"] = num_likes - num_dislikes
            else:
                formatted_post["num_likes"] = num_likes
                formatted_post["num_dislikes"] = num_dislikes
            posts.append(formatted_post)
        return posts

    def _root_post(self, post_id: int) -> tuple[int, int]:
        """Return (root_post_id, root_author_id), guarding malformed cycles."""
        current = post_id
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            row = self.db.execute(
                "SELECT user_id, original_post_id FROM post WHERE post_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                raise LookupError(f"post {current} does not exist")
            author_id, parent_id = int(row[0]), row[1]
            if parent_id is None:
                return current, author_id
            current = int(parent_id)
        raise ValueError(f"cycle detected in original_post_id chain for {post_id}")

    def _maybe_update_controller(self, timestep: int) -> None:
        if self.controller is None or self._last_control_timestep == timestep:
            return
        if self.intervention_budget is None:
            raise ValueError("a controller requires an explicit intervention budget")
        if self.state_provider is not None:
            state = self.state_provider(timestep)
        else:
            state = state_from_database(
                self.db,
                self.run_id,
                timestep,
                self.controller_window,
                len(set(self.user_communities.values())),
                self._cumulative_intervention_cost,
                self.intervention_budget,
            )
        action = self.controller.update(
            timestep, state, self.policy.omega_intra, self.policy.omega_inter
        )
        self.policy.update(action.omega_intra, action.omega_inter)
        self._cumulative_intervention_cost += action.expected_cost
        self._last_control_timestep = timestep
        logger.info("controller update at %s: %s", timestep, action)

    async def update_rec_table(self) -> dict[str, Any]:
        """Refresh OASIS' base candidate cache; governance happens at exposure."""
        await super().update_rec_table()
        count_row = self.db.execute("SELECT COUNT(*) FROM rec").fetchone()
        return {"success": True, "candidate_count": int(count_row[0])}

    async def refresh(self, agent_id: int) -> dict[str, Any]:
        """Generate, govern, log, and return the final feed for one user."""
        timestep = self._timestep()
        self._maybe_update_controller(timestep)
        user_id = int(agent_id)
        if user_id not in self.user_communities:
            return {"success": False, "error": f"missing community for user {user_id}"}
        user_community = self.user_communities[user_id]

        candidates = self.candidate_generator.generate(
            self.db,
            user_id,
            self.refresh_rec_post_count,
            self.following_post_count,
        )
        shown_ids: list[int] = []
        records: list[ImpressionRecord] = []
        try:
            for candidate in candidates:
                root_id, root_author = self._root_post(candidate.post_id)
                if root_author not in self.user_communities:
                    raise KeyError(f"missing community for author {root_author}")
                author_community = self.user_communities[root_author]
                if root_id in self.post_risk_scores:
                    risk = self.post_risk_scores[root_id]
                elif candidate.post_id in self.post_risk_scores:
                    risk = self.post_risk_scores[candidate.post_id]
                else:
                    raise KeyError(
                        f"missing risk score for root post {root_id} "
                        f"(candidate {candidate.post_id})"
                    )
                decision = self.policy.decide(
                    user_community,
                    author_community,
                    risk,
                    decision_key=hash((timestep, user_id, root_id)),
                )
                records.append(
                    ImpressionRecord(
                        run_id=self.run_id,
                        timestep=timestep,
                        user_id=user_id,
                        post_id=candidate.post_id,
                        root_post_id=root_id,
                        user_community=user_community,
                        author_community=author_community,
                        base_score=candidate.base_score,
                        risk_score=risk,
                        omega_intra=self.policy.omega_intra,
                        omega_inter=self.policy.omega_inter,
                        keep_probability=decision.keep_probability,
                        shown=decision.shown,
                        source=candidate.source,
                    )
                )
                if decision.shown:
                    shown_ids.append(candidate.post_id)
        except (KeyError, LookupError, ValueError) as exc:
            return {"success": False, "error": str(exc)}

        self.impressions.append_many(records)
        if not shown_ids:
            result: dict[str, Any] = {
                "success": True,
                "message": "No posts passed exposure policy.",
                "posts": [],
            }
        else:
            placeholders = ", ".join("?" for _ in shown_ids)
            rows = self.db.execute(
                f"""
                SELECT post_id, user_id, original_post_id, content,
                       quote_content, created_at, num_likes, num_dislikes,
                       num_shares
                FROM post WHERE post_id IN ({placeholders})
                """,
                shown_ids,
            ).fetchall()
            by_id = {int(row[0]): row for row in rows}
            ordered = [by_id[post_id] for post_id in shown_ids if post_id in by_id]
            result = {
                "success": True,
                "posts": self._format_feed_posts(ordered),
            }
        self.pl_utils._record_trace(
            user_id,
            ActionType.REFRESH.value,
            {"posts": result.get("posts", [])},
            self._current_time(),
        )
        return result
