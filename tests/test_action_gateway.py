from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from adaptive_diffusionguard.llm.action_gateway import (
    ActionDecisionError,
    ActionDecisionGateway,
    attach_action_gateway,
)
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    ActionMaskBuilder,
    StructuredActionResponse,
    build_decision_messages,
    build_response_format,
)
from training.build_dataset import examples_from_oasis
from training.dataset_pipeline import DatasetIntegrityError


class FakeSocialAction:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail = False

    async def refresh(self) -> dict[str, Any]:
        return {"success": True, "posts": [{"post_id": 7}, {"post_id": 9}]}

    def _result(self) -> dict[str, Any]:
        return {"success": not self.fail}

    async def repost(self, post_id: int) -> dict[str, Any]:
        self.calls.append(("repost", post_id))
        return self._result()

    async def quote_post(self, post_id: int, text: str) -> dict[str, Any]:
        self.calls.append(("quote_post", post_id, text))
        return self._result()

    async def report_post(self, post_id: int, text: str) -> dict[str, Any]:
        self.calls.append(("report_post", post_id, text))
        return self._result()

    async def do_nothing(self) -> dict[str, Any]:
        self.calls.append(("do_nothing",))
        return self._result()


def ready_gateway() -> tuple[ActionDecisionGateway, FakeSocialAction]:
    action = FakeSocialAction()
    gateway = ActionDecisionGateway(action)
    gateway.update_visible_feed(
        {"success": True, "posts": [{"post_id": 7}, {"post_id": 9}]}
    )
    return gateway, action


@pytest.mark.parametrize(
    ("choice_id", "rationale", "expected_call"),
    [
        ("repost:7", "share", ("repost", 7)),
        ("quote:7", " context ", ("quote_post", 7, "context")),
        ("report:9", " unverified ", ("report_post", 9, "unverified")),
        ("ignore", "nothing relevant", ("do_nothing",)),
    ],
)
def test_four_choices_dispatch_deterministically(
    choice_id: str, rationale: str, expected_call: tuple[Any, ...]
) -> None:
    gateway, action = ready_gateway()

    result = asyncio.run(
        gateway.dispatch_choice(
            choice_id, rationale, ["ignore", "repost:7", "quote:7", "report:9"]
        )
    )

    assert result["success"] is True
    assert action.calls == [expected_call]
    assert gateway.audit.dispatcher_success_count == 1


def test_illegal_choice_and_invisible_post_never_dispatch() -> None:
    gateway, action = ready_gateway()
    with pytest.raises(ActionDecisionError, match="not currently legal"):
        asyncio.run(gateway.dispatch_choice("repost:99", "x", ["ignore"]))
    with pytest.raises(ActionDecisionError, match="not in the current feed"):
        asyncio.run(gateway.dispatch_choice("repost:99", "x", ["repost:99"]))
    assert action.calls == []
    assert gateway.audit.invalid_choice_count == 1
    assert gateway.audit.invalid_post_reference_count == 1


def test_dispatch_failure_is_explicit() -> None:
    gateway, action = ready_gateway()
    action.fail = True
    with pytest.raises(ActionDecisionError, match="dispatch failed"):
        asyncio.run(gateway.dispatch_choice("ignore", "x", ["ignore"]))
    assert gateway.audit.dispatcher_failure_count == 1
    assert gateway.audit.dispatcher_success_count == 0


def test_attach_gateway_removes_all_model_tools() -> None:
    action = FakeSocialAction()
    agent = SimpleNamespace(
        env=SimpleNamespace(action=action),
        tool_dict={"refresh": object(), "repost": object()},
        action_tools=[object()],
    )
    gateway = attach_action_gateway(agent)
    assert isinstance(gateway, ActionDecisionGateway)
    assert agent.tool_dict == {}
    assert agent.action_tools == []


def database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            original_post_id INTEGER,
            quote_content TEXT
        );
        CREATE TABLE report (user_id INTEGER NOT NULL, post_id INTEGER NOT NULL);
        INSERT INTO post VALUES (1, 1, NULL, NULL);
        INSERT INTO post VALUES (2, 2, NULL, NULL);
        """
    )
    return connection


def test_action_mask_is_dynamic_and_uses_persisted_oasis_rules() -> None:
    connection = database()
    builder = ActionMaskBuilder(connection)
    initial = builder.build(9, frozenset({1, 2, 999}))
    assert initial.choices == (
        "ignore",
        "repost:1",
        "quote:1",
        "report:1",
        "repost:2",
        "quote:2",
        "report:2",
    )
    assert all("999" not in choice for choice in initial.choices)

    connection.execute("INSERT INTO post VALUES (10, 9, 1, NULL)")
    connection.execute("INSERT INTO report VALUES (9, 1)")
    connection.execute("INSERT INTO post VALUES (11, 9, 1, 'an earlier quote')")
    changed = builder.build(9, frozenset({1, 2}))
    assert "repost:1" not in changed.choices
    assert "report:1" not in changed.choices
    # OASIS 0.2.5 explicitly allows repeated quotes with different text.
    assert "quote:1" in changed.choices
    assert changed.choices[0] == "ignore"


def test_repost_of_visible_repost_checks_root_state() -> None:
    connection = database()
    connection.execute("INSERT INTO post VALUES (3, 2, 1, NULL)")
    connection.execute("INSERT INTO post VALUES (10, 9, 1, NULL)")
    choices = ActionMaskBuilder(connection).build(9, frozenset({3})).choices
    assert "repost:3" not in choices
    assert "quote:3" in choices
    assert "report:3" in choices


def test_strict_response_format_and_messages_have_no_tool_interface() -> None:
    mask = ActionMask(("ignore", "quote:7"))
    response_format = build_response_format(mask)
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert schema["properties"]["choice_id"]["enum"] == ["ignore", "quote:7"]
    assert schema["required"] == ["choice_id", "rationale"]
    assert schema["additionalProperties"] is False
    assert "nullable" not in str(response_format)

    messages = build_decision_messages(
        profile="synthetic", feed={"posts": [{"post_id": 7}]}, mask=mask
    )
    assert {message["role"] for message in messages} == {"system", "user"}
    assert "tools" not in str(messages)
    assert "tool_choice" not in str(messages)


def test_structured_response_forbids_extra_fields_and_empty_rationale() -> None:
    assert (
        StructuredActionResponse.model_validate(
            {"choice_id": "ignore", "rationale": "audit reason"}
        ).choice_id
        == "ignore"
    )
    with pytest.raises(ValidationError):
        StructuredActionResponse.model_validate(
            {"choice_id": "ignore", "rationale": "", "extra": True}
        )


def test_failed_or_illegal_dispatch_does_not_enter_teacher_jsonl(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trace.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE trace (
                user_id INTEGER,
                created_at TEXT,
                action TEXT,
                info TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO trace VALUES (?, ?, ?, ?)",
            (9, "2026-01-01", "do_nothing", json.dumps({"success": True})),
        )

    gateway, action = ready_gateway()
    action.fail = True
    with pytest.raises(ActionDecisionError):
        asyncio.run(gateway.dispatch_choice("ignore", "audit", ["ignore"]))
    with pytest.raises(ActionDecisionError):
        asyncio.run(gateway.dispatch_choice("report:999", "bad", ["ignore"]))

    with pytest.raises(DatasetIntegrityError, match="lacks native decision snapshots"):
        list(
            examples_from_oasis(
                db_path,
                {"9": "synthetic"},
                {"9": "community-0"},
                "teacher_synthetic",
            )
        )
    assert not (tmp_path / "teacher.jsonl").exists()
