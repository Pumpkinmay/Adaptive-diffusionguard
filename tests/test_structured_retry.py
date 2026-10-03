from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from camel.models import BaseModelBackend

from adaptive_diffusionguard.llm.model_factory import LLMSettings
from adaptive_diffusionguard.llm.runtime import (
    LLMRuntime,
    ManagedModelBackend,
    ResponseCache,
)
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    ActionMaskBuilder,
    DecisionSnapshot,
    build_response_format,
    revalidate_snapshot_choice,
)
from adaptive_diffusionguard.llm.structured_retry import (
    CORRECTION_PROMPT,
    request_with_structured_correction,
)


class ProviderFailure(Exception):
    def __init__(self, status_code: int, code: str, *, secret: str = "") -> None:
        super().__init__(f"provider failure {status_code} {code}")
        self.status_code = status_code
        self.body = {
            "code": code,
            "failed_generation": f"do-not-copy-{secret}",
        }


class FakeTokenCounter:
    def count_tokens_from_messages(self, messages: list[Any]) -> int:
        return len(messages)


class SequenceStructuredBackend(BaseModelBackend):
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self._counter = FakeTokenCounter()
        super().__init__(
            model_type="fake-structured-model",
            model_config_dict={"temperature": 0.0, "max_tokens": 512},
        )

    @property
    def token_counter(self) -> FakeTokenCounter:
        return self._counter

    def run_strict_structured(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
    ) -> Any:
        self.requests.append(
            {"messages": messages, "response_format": response_format}
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def _run(self, messages, response_format=None, tools=None):
        raise AssertionError("generic/tool path must not be used")

    async def _arun(self, messages, response_format=None, tools=None):
        raise AssertionError("generic/tool path must not be used")


def response(choice_id: str = "ignore") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"choice_id": choice_id, "rationale": "safe rationale"}
                    )
                },
            }
        ]
    }


def managed_model(
    tmp_path: Path,
    outcomes: list[Any],
    *,
    max_calls: int = 30,
    cache: bool = False,
    secret: str = "unit-test-secret",
) -> tuple[ManagedModelBackend, SequenceStructuredBackend]:
    settings = LLMSettings(
        enabled=True,
        provider="groq",
        llm_model="fake-structured-model",
        groq_api_key=secret,
        temperature=0.0,
        max_tokens=512,
        max_retries=0,
        max_concurrency=1,
        max_calls_per_run=max_calls,
        request_interval_seconds=0.0,
        cache_enabled=cache,
        cache_path=tmp_path / "cache.sqlite3",
    )
    backend = SequenceStructuredBackend(outcomes)
    runtime = LLMRuntime(
        settings,
        cache=ResponseCache(settings.cache_path) if cache else None,
        model_config=backend.model_config_dict,
        sleeper=lambda _: None,
    )
    return ManagedModelBackend(backend, runtime), backend


def decision_snapshot() -> DecisionSnapshot:
    mask = ActionMask(("ignore", "report:1"))
    return DecisionSnapshot.capture(
        user_id=7,
        timestep=2,
        feed={"success": True, "posts": [{"post_id": 1, "content": "safe"}]},
        visible_post_ids=frozenset({1}),
        mask=mask,
        response_format=build_response_format(mask),
        messages=[{"role": "user", "content": "original safe context"}],
        state_identifier="state-before-first-attempt",
    )


@pytest.mark.asyncio
async def test_json_validate_failed_retries_once_and_succeeds(
    tmp_path: Path,
) -> None:
    secret = "gsk_UNIT_TEST_ONLY"
    model, backend = managed_model(
        tmp_path,
        [ProviderFailure(400, "json_validate_failed", secret=secret), response()],
        cache=True,
        secret=secret,
    )
    snapshot = decision_snapshot()
    retry_messages = [
        *snapshot.messages_value(),
        {"role": "user", "content": CORRECTION_PROMPT},
    ]
    assert model.runtime.store_response(
        response(),
        messages=retry_messages,
        response_format=snapshot.response_format_value(),
        model_config=model.backend.model_config_dict,
    )

    result = await request_with_structured_correction(model, snapshot)
    model.runtime.record_structured_completion(retried=result.retried)

    assert result.retried is True
    assert len(backend.requests) == 2
    assert backend.requests[1]["messages"][:-1] == backend.requests[0]["messages"]
    assert backend.requests[1]["messages"][-1] == {
        "role": "user",
        "content": CORRECTION_PROMPT,
    }
    assert secret not in json.dumps(backend.requests[1])
    assert "failed_generation" not in json.dumps(backend.requests[1])
    assert backend.requests[1]["response_format"] == backend.requests[0][
        "response_format"
    ]
    assert result.request.context is snapshot
    assert snapshot.feed_value() == {
        "posts": [{"content": "safe", "post_id": 1}],
        "success": True,
    }
    retry_enum = backend.requests[1]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["choice_id"]["enum"]
    assert retry_enum == list(snapshot.legal_choice_ids)
    stats = model.runtime.stats.to_dict()
    assert stats["physical_remote_attempts"] == 2
    assert stats["first_attempt_count"] == 1
    assert stats["first_attempt_success_count"] == 0
    assert stats["json_validate_failed_count"] == 1
    assert stats["structured_retry_attempt_count"] == 1
    assert stats["structured_retry_success_count"] == 1
    assert stats["structured_retry_failure_count"] == 0
    assert stats["unrecovered_error_count"] == 0
    assert stats["post_retry_success_rate"] == 1.0
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        keys = [row[0] for row in connection.execute(
            "SELECT cache_key FROM llm_response_cache"
        )]
    assert all(secret not in key for key in keys)


@pytest.mark.asyncio
async def test_second_json_validate_failed_stops_after_one_retry(
    tmp_path: Path,
) -> None:
    model, backend = managed_model(
        tmp_path,
        [
            ProviderFailure(400, "json_validate_failed"),
            ProviderFailure(400, "json_validate_failed"),
        ],
        cache=True,
        max_calls=10,
    )

    with pytest.raises(Exception, match="json_validate_failed"):
        await request_with_structured_correction(model, decision_snapshot())

    assert len(backend.requests) == 2
    stats = model.runtime.stats.to_dict()
    assert stats["physical_remote_attempts"] == 2
    assert stats["json_validate_failed_count"] == 2
    assert stats["structured_retry_attempt_count"] == 1
    assert stats["structured_retry_success_count"] == 0
    assert stats["structured_retry_failure_count"] == 1
    assert stats["unrecovered_error_count"] == 1
    assert stats["unrecovered_http_statuses"] == {"400": 1}
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM llm_response_cache"
        ).fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (ProviderFailure(400, "other_bad_request"), "400"),
        (ProviderFailure(401, "unauthorized"), "401"),
        (ProviderFailure(403, "forbidden"), "403"),
        (ProviderFailure(429, "rate_limit"), "429"),
        (ProviderFailure(500, "server_error"), "500"),
        (TimeoutError("timeout"), None),
    ],
)
async def test_other_provider_failures_never_use_structured_retry(
    tmp_path: Path,
    failure: BaseException,
    expected_status: str | None,
) -> None:
    model, backend = managed_model(tmp_path, [failure])

    with pytest.raises(type(failure)):
        await request_with_structured_correction(model, decision_snapshot())

    assert len(backend.requests) == 1
    stats = model.runtime.stats.to_dict()
    assert stats["structured_retry_attempt_count"] == 0
    assert stats["unrecovered_error_count"] == 1
    expected = {expected_status: 1} if expected_status is not None else {}
    assert stats["unrecovered_http_statuses"] == expected


@pytest.mark.asyncio
async def test_no_remaining_call_budget_prevents_structured_retry(
    tmp_path: Path,
) -> None:
    model, backend = managed_model(
        tmp_path,
        [ProviderFailure(400, "json_validate_failed")],
        max_calls=1,
    )

    with pytest.raises(Exception, match="json_validate_failed"):
        await request_with_structured_correction(model, decision_snapshot())

    assert len(backend.requests) == 1
    assert model.runtime.stats.structured_retry_attempt_count == 0
    assert model.runtime.stats.remote_calls == 1


@pytest.mark.asyncio
async def test_one_remaining_call_allows_exactly_one_retry(
    tmp_path: Path,
) -> None:
    model, backend = managed_model(
        tmp_path,
        [
            ProviderFailure(400, "json_validate_failed"),
            ProviderFailure(400, "json_validate_failed"),
        ],
        max_calls=2,
    )

    with pytest.raises(Exception, match="json_validate_failed"):
        await request_with_structured_correction(model, decision_snapshot())

    assert len(backend.requests) == 2
    assert model.runtime.stats.remote_calls == 2
    assert model.runtime.stats.structured_retry_attempt_count == 1
    assert model.runtime.remaining_calls == 0


@pytest.mark.asyncio
async def test_state_change_after_retry_rejects_dispatch_without_third_request(
    tmp_path: Path,
) -> None:
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
        """
    )
    builder = ActionMaskBuilder(connection)
    mask = builder.build(7, frozenset({1}))
    snapshot = DecisionSnapshot.capture(
        user_id=7,
        timestep=1,
        feed={"success": True, "posts": [{"post_id": 1}]},
        visible_post_ids=frozenset({1}),
        mask=mask,
        response_format=build_response_format(mask),
        messages=[{"role": "user", "content": "original"}],
        state_identifier=builder.state_identifier(7, frozenset({1})),
    )
    model, backend = managed_model(
        tmp_path,
        [
            ProviderFailure(400, "json_validate_failed"),
            response("report:1"),
        ],
    )

    result = await request_with_structured_correction(model, snapshot)
    connection.execute("INSERT INTO report VALUES (7, 1)")

    with pytest.raises(ValueError, match="no longer legal"):
        revalidate_snapshot_choice(snapshot, "report:1", builder)
    assert result.retried is True
    assert len(backend.requests) == 2
    assert model.runtime.stats.remote_calls == 2
    assert connection.execute("SELECT COUNT(*) FROM report").fetchone()[0] == 1
