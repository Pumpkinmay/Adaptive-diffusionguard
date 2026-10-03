"""Dynamic legal-action masks and strict structured response validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

from .actions import extract_response_text

NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]


class StructuredActionResponse(BaseModel):
    """Only structured model response accepted by the local dispatcher."""

    model_config = ConfigDict(extra="forbid", strict=True)

    choice_id: str
    rationale: NonEmptyText


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Stable legal choices derived from one freshly observed feed."""

    choices: tuple[str, ...]

    @property
    def has_platform_action(self) -> bool:
        return len(self.choices) > 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Immutable first-exposure context for one logical decision.

    JSON-shaped values are stored as canonical bytes so neither the caller nor
    the retry path can mutate the original feed, schema, or model messages.
    Accessors always return fresh decoded values.
    """

    user_id: int
    timestep: int
    feed: bytes
    visible_post_ids: tuple[int, ...]
    legal_choice_ids: tuple[str, ...]
    response_format: bytes
    messages: bytes
    state_identifier: str

    @classmethod
    def capture(
        cls,
        *,
        user_id: int,
        timestep: int,
        feed: dict[str, Any],
        visible_post_ids: set[int] | frozenset[int],
        mask: ActionMask,
        response_format: dict[str, Any],
        messages: list[dict[str, Any]],
        state_identifier: str,
    ) -> DecisionSnapshot:
        return cls(
            user_id=user_id,
            timestep=timestep,
            feed=_canonical_json_bytes(feed),
            visible_post_ids=tuple(sorted(visible_post_ids)),
            legal_choice_ids=tuple(mask.choices),
            response_format=_canonical_json_bytes(response_format),
            messages=_canonical_json_bytes(messages),
            state_identifier=state_identifier,
        )

    def feed_value(self) -> dict[str, Any]:
        return json.loads(self.feed)

    def response_format_value(self) -> dict[str, Any]:
        return json.loads(self.response_format)

    def messages_value(self) -> list[dict[str, Any]]:
        return json.loads(self.messages)


class ActionMaskBuilder:
    """Mirror OASIS action legality using its persisted SQLite state.

    OASIS 0.2.5 disallows a repeated repost by the same user for the target
    post (and for the root when the visible target is itself a repost), and
    disallows a repeated report for the exact visible post. Its quote_post
    implementation explicitly permits repeated quotes because their text may
    differ, and the post/report schemas add no uniqueness constraint.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def build(
        self, user_id: int, visible_post_ids: set[int] | frozenset[int]
    ) -> ActionMask:
        choices = ["ignore"]
        for post_id in sorted(visible_post_ids):
            row = self.connection.execute(
                "SELECT original_post_id, quote_content FROM post WHERE post_id = ?",
                (post_id,),
            ).fetchone()
            if row is None:
                continue
            original_post_id, quote_content = row
            if self._can_repost(user_id, post_id, original_post_id, quote_content):
                choices.append(f"repost:{post_id}")
            # OASIS quote_post explicitly allows repeated quotes.
            choices.append(f"quote:{post_id}")
            if not self._already_reported(user_id, post_id):
                choices.append(f"report:{post_id}")
        return ActionMask(tuple(choices))

    def state_identifier(
        self, user_id: int, visible_post_ids: set[int] | frozenset[int]
    ) -> str:
        """Fingerprint persisted rows that determine this user's legal mask."""
        post_ids = sorted(visible_post_ids)
        if not post_ids:
            material = {"user_id": user_id, "posts": [], "reports": [], "reposts": []}
        else:
            placeholders = ", ".join("?" for _ in post_ids)
            posts = self.connection.execute(
                f"SELECT post_id, user_id, original_post_id, quote_content "
                f"FROM post WHERE post_id IN ({placeholders}) ORDER BY post_id",
                post_ids,
            ).fetchall()
            reports = self.connection.execute(
                f"SELECT post_id FROM report WHERE user_id = ? "
                f"AND post_id IN ({placeholders}) ORDER BY post_id",
                (user_id, *post_ids),
            ).fetchall()
            reposts = self.connection.execute(
                "SELECT original_post_id FROM post WHERE user_id = ? "
                "AND original_post_id IS NOT NULL AND quote_content IS NULL "
                "ORDER BY original_post_id",
                (user_id,),
            ).fetchall()
            material = {
                "user_id": user_id,
                "posts": posts,
                "reports": reports,
                "reposts": reposts,
            }
        return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()

    def _can_repost(
        self,
        user_id: int,
        post_id: int,
        original_post_id: int | None,
        quote_content: str | None,
    ) -> bool:
        targets = [post_id]
        is_repost = original_post_id is not None and quote_content is None
        if is_repost:
            targets.append(self._root_post_id(post_id))
        placeholders = ", ".join("?" for _ in targets)
        row = self.connection.execute(
            f"SELECT 1 FROM post WHERE user_id = ? "
            f"AND original_post_id IN ({placeholders}) LIMIT 1",
            (user_id, *targets),
        ).fetchone()
        return row is None

    def _already_reported(self, user_id: int, post_id: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM report WHERE user_id = ? AND post_id = ? LIMIT 1",
                (user_id, post_id),
            ).fetchone()
            is not None
        )

    def _root_post_id(self, post_id: int) -> int:
        current = post_id
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            row = self.connection.execute(
                "SELECT original_post_id FROM post WHERE post_id = ?", (current,)
            ).fetchone()
            if row is None or row[0] is None:
                return current
            current = int(row[0])
        raise ValueError(f"cycle detected while resolving post {post_id}")


def build_response_format(mask: ActionMask) -> dict[str, Any]:
    """Build the exact Groq strict JSON-schema request object."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "adaptive_diffusionguard_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice_id": {
                        "type": "string",
                        "enum": list(mask.choices),
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["choice_id", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def revalidate_snapshot_choice(
    snapshot: DecisionSnapshot,
    choice_id: str,
    mask_builder: ActionMaskBuilder,
) -> ActionMask:
    """Reject a choice if the original or current persisted state disallows it."""
    if choice_id not in snapshot.legal_choice_ids:
        raise ValueError("choice_id is not in the original decision snapshot")
    current = mask_builder.build(
        snapshot.user_id, frozenset(snapshot.visible_post_ids)
    )
    if choice_id not in current.choices:
        raise ValueError("choice_id is no longer legal in current OASIS state")
    return current


def build_decision_messages(
    *, profile: str, feed: dict[str, Any], mask: ActionMask
) -> list[dict[str, str]]:
    system = (
        "You are a synthetic social-media behavior teacher. Select exactly one "
        "choice_id from the allowed enum supplied by the response schema. Give "
        "a concise rationale. Do not output fields outside the schema. These are "
        "behavior-distillation labels, not observed human behavior."
    )
    user = json.dumps(
        {
            "profile": profile,
            "visible_feed": feed.get("posts", []),
            "legal_choice_ids": list(mask.choices),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_structured_response(response: Any) -> StructuredActionResponse:
    """Strictly parse assistant content; never synthesize an ignore fallback."""
    choices = response.get("choices") if isinstance(response, dict) else getattr(
        response, "choices", None
    )
    if not choices:
        raise ValueError("structured response contains no choices")
    choice = choices[0]
    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    if finish_reason == "length":
        raise ValueError("structured response was truncated")
    message = (
        choice.get("message")
        if isinstance(choice, dict)
        else getattr(choice, "message", None)
    )
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if tool_calls:
        raise ValueError("structured response unexpectedly contains tool calls")
    raw = extract_response_text(response)
    return StructuredActionResponse.model_validate_json(raw)
