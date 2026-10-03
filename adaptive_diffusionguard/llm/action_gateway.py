"""Local, auditable dispatch from structured choices to OASIS actions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ActionName = Literal["repost", "quote", "report", "ignore"]
ACTION_NAMES: tuple[ActionName, ...] = ("repost", "quote", "report", "ignore")


class ActionDecisionError(ValueError):
    """A payload-free error for rejected or failed local decisions."""


class SocialActionProtocol(Protocol):
    async def refresh(self) -> dict[str, Any]: ...

    async def repost(self, post_id: int) -> dict[str, Any]: ...

    async def quote_post(self, post_id: int, text: str) -> dict[str, Any]: ...

    async def report_post(self, post_id: int, text: str) -> dict[str, Any]: ...

    async def do_nothing(self) -> dict[str, Any]: ...


def _empty_action_counts() -> dict[str, int]:
    return {name: 0 for name in ACTION_NAMES}


@dataclass(slots=True)
class GatewayAudit:
    """Non-sensitive local dispatcher counters."""

    dispatcher_success_count: int = 0
    dispatcher_failure_count: int = 0
    invalid_choice_count: int = 0
    invalid_post_reference_count: int = 0
    action_counts: dict[str, int] = field(default_factory=_empty_action_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatcher_success_count": self.dispatcher_success_count,
            "dispatcher_failure_count": self.dispatcher_failure_count,
            "invalid_choice_count": self.invalid_choice_count,
            "invalid_post_reference_count": self.invalid_post_reference_count,
            "action_counts": {
                name: int(self.action_counts.get(name, 0))
                for name in ACTION_NAMES
            },
        }

    @classmethod
    def merge(cls, audits: Iterable[GatewayAudit]) -> GatewayAudit:
        merged = cls()
        for audit in audits:
            merged.dispatcher_success_count += audit.dispatcher_success_count
            merged.dispatcher_failure_count += audit.dispatcher_failure_count
            merged.invalid_choice_count += audit.invalid_choice_count
            merged.invalid_post_reference_count += (
                audit.invalid_post_reference_count
            )
            for name in ACTION_NAMES:
                merged.action_counts[name] += audit.action_counts.get(name, 0)
        return merged


class ActionDecisionGateway:
    """Revalidate and deterministically dispatch one structured choice."""

    def __init__(
        self,
        social_action: SocialActionProtocol,
        audit: GatewayAudit | None = None,
    ) -> None:
        self._social_action = social_action
        self.audit = audit or GatewayAudit()
        self._visible_post_ids: set[int] = set()

    @property
    def visible_post_ids(self) -> frozenset[int]:
        return frozenset(self._visible_post_ids)

    def update_visible_feed(self, refresh_response: Any) -> None:
        """Replace visibility with exactly the posts in the latest feed."""
        visible: set[int] = set()
        if isinstance(refresh_response, Mapping) and refresh_response.get("success"):
            posts = refresh_response.get("posts")
            if isinstance(posts, list):
                for post in posts:
                    if not isinstance(post, Mapping):
                        continue
                    post_id = post.get("post_id")
                    if (
                        isinstance(post_id, int)
                        and not isinstance(post_id, bool)
                        and post_id > 0
                    ):
                        visible.add(post_id)
        self._visible_post_ids = visible

    async def dispatch_choice(
        self,
        choice_id: str,
        rationale: str,
        legal_choices: Iterable[str],
    ) -> dict[str, Any]:
        """Dispatch only a currently legal structured choice."""
        legal = frozenset(legal_choices)
        if choice_id not in legal:
            self.audit.invalid_choice_count += 1
            raise ActionDecisionError("choice_id is not currently legal")

        action, post_id = self._parse_choice(choice_id)
        if post_id is not None and post_id not in self._visible_post_ids:
            self.audit.invalid_post_reference_count += 1
            raise ActionDecisionError("choice post_id is not in the current feed")

        rationale = rationale.strip()
        if action in {"quote", "report"} and not rationale:
            self.audit.dispatcher_failure_count += 1
            raise ActionDecisionError(f"{action} requires a non-empty rationale")

        if action == "repost":
            result = await self._social_action.repost(post_id)
        elif action == "quote":
            result = await self._social_action.quote_post(post_id, rationale)
        elif action == "report":
            result = await self._social_action.report_post(post_id, rationale)
        else:
            result = await self._social_action.do_nothing()

        if not isinstance(result, dict) or result.get("success") is not True:
            self.audit.dispatcher_failure_count += 1
            raise ActionDecisionError(f"OASIS {action} dispatch failed")
        self.audit.dispatcher_success_count += 1
        self.audit.action_counts[action] += 1
        return result

    def _parse_choice(self, choice_id: str) -> tuple[ActionName, int | None]:
        if choice_id == "ignore":
            return "ignore", None
        try:
            raw_action, raw_post_id = choice_id.split(":", 1)
            post_id = int(raw_post_id)
        except (TypeError, ValueError):
            self.audit.invalid_choice_count += 1
            raise ActionDecisionError("choice_id encoding is invalid") from None
        if raw_action not in {"repost", "quote", "report"} or post_id <= 0:
            self.audit.invalid_choice_count += 1
            raise ActionDecisionError("choice_id encoding is invalid")
        return raw_action, post_id  # type: ignore[return-value]


def attach_action_gateway(agent: Any) -> ActionDecisionGateway:
    """Attach local dispatch and ensure the model has no callable tools."""
    social_action = agent.env.action
    gateway = ActionDecisionGateway(social_action)
    agent.tool_dict.clear()
    agent.action_tools = []
    return gateway
