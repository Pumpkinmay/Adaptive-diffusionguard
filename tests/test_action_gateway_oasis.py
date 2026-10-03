from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from camel.models import BaseModelBackend
from camel.types import ChatCompletion

from adaptive_diffusionguard.experiments.real_llm_validation import run_validation
from adaptive_diffusionguard.llm.model_factory import LLMSettings
from adaptive_diffusionguard.llm.runtime import (
    LLMRuntime,
    ManagedModelBackend,
    ResponseCache,
)
from adaptive_diffusionguard.llm.structured_retry import CORRECTION_PROMPT


class FakeTokenCounter:
    def count_tokens_from_messages(self, messages: list[Any]) -> int:
        return len(messages)

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class StrictStructuredFakeBackend(BaseModelBackend):
    """Local backend whose narrow signature proves tools are not supplied."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.before_request: Callable[[], None] | None = None
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
    ) -> ChatCompletion:
        if self.before_request is not None:
            self.before_request()
        self.requests.append({"messages": messages, "response_format": response_format})
        return self._response(response_format)

    def _response(self, response_format: dict[str, Any]) -> ChatCompletion:
        choices = response_format["json_schema"]["schema"]["properties"]["choice_id"][
            "enum"
        ]
        desired = ("repost", "quote", "report", "ignore")[(len(self.requests) - 1) % 4]
        if desired == "ignore":
            choice_id = "ignore"
        else:
            choice_id = next(
                (value for value in choices if value.startswith(f"{desired}:")),
                "ignore",
            )
        return ChatCompletion.model_validate(
            {
                "id": f"fake-{len(self.requests)}",
                "object": "chat.completion",
                "created": len(self.requests),
                "model": "fake-structured-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "choice_id": choice_id,
                                    "rationale": "synthetic audit rationale",
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    def _run(self, messages, response_format=None, tools=None):
        raise AssertionError("the structured path must not call the generic backend")

    async def _arun(self, messages, response_format=None, tools=None):
        raise AssertionError("the structured path must not call the generic backend")


class JsonValidateFailed(Exception):
    status_code = 400

    def __init__(self) -> None:
        super().__init__("provider rejected generated structured output")
        self.body = {
            "code": "json_validate_failed",
            "failed_generation": "must-never-enter-retry-or-output",
        }


class CorrectingStructuredFakeBackend(StrictStructuredFakeBackend):
    def __init__(
        self,
        *,
        fail_first_decisions: set[int],
        fail_retry_decisions: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.fail_first_decisions = fail_first_decisions
        self.fail_retry_decisions = fail_retry_decisions or set()
        self.logical_decision = 0
        self.pending_retry_decision: int | None = None

    def run_strict_structured(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
    ) -> ChatCompletion:
        self.requests.append({"messages": messages, "response_format": response_format})
        is_retry = messages[-1].get("content") == CORRECTION_PROMPT
        if is_retry:
            decision = self.pending_retry_decision
            self.pending_retry_decision = None
            if decision in self.fail_retry_decisions:
                raise JsonValidateFailed()
        else:
            self.logical_decision += 1
            decision = self.logical_decision
            if decision in self.fail_first_decisions:
                self.pending_retry_decision = decision
                raise JsonValidateFailed()
        return self._response(response_format)


@pytest.mark.asyncio
async def test_real_oasis_five_by_three_with_fake_structured_backend(
    tmp_path: Path,
) -> None:
    settings = LLMSettings(
        enabled=True,
        provider="groq",
        llm_model="fake-structured-model",
        groq_api_key="unit-test-only-secret",
        temperature=0.0,
        max_tokens=512,
        max_retries=0,
        max_concurrency=1,
        max_calls_per_run=30,
        request_interval_seconds=0.0,
        cache_enabled=True,
        cache_path=tmp_path / "responses.sqlite3",
    )
    backend = StrictStructuredFakeBackend()
    runtime = LLMRuntime(
        settings,
        cache=ResponseCache(settings.cache_path),
        model_config=backend.model_config_dict,
        sleeper=lambda _: None,
    )
    managed = ManagedModelBackend(backend, runtime)
    output = tmp_path / "fake-structured-validation"
    pending_observations: list[tuple[int, int]] = []

    def observe_pending_snapshot() -> None:
        database = output / "real_llm_validation.db"
        with sqlite3.connect(database) as connection:
            total, pending = connection.execute(
                "SELECT COUNT(*), SUM(status = 'pending') "
                "FROM diffusionguard_decision_snapshot"
            ).fetchone()
        pending_observations.append((int(total), int(pending or 0)))

    backend.before_request = observe_pending_snapshot

    summary = await run_validation(settings, output, model_override=managed)

    assert summary["status"] == "success"
    assert summary["validation_mode"] == "strict_structured_action_mask"
    assert summary["logical_decisions"] == 15
    assert summary["completed_decisions"] == 15
    assert summary["structured_schema_valid_count"] == 15
    assert summary["legal_choice_count"] == 15
    assert summary["dispatcher_success_count"] == 15
    assert summary["teacher_examples"] == 15
    assert summary["tool_call_count"] == 0
    assert summary["raw_tool_call_count"] == 0
    assert summary["no_tool_call_response_count"] == 0
    assert summary["provider_tool_use_failed_count"] == 0
    assert summary["secret_scan_passed"] is True
    assert sum(summary["action_counts"].values()) == 15
    assert all(
        summary["action_counts"][name] > 0
        for name in ("repost", "quote", "report", "ignore")
    )
    assert pending_observations == [(index, 1) for index in range(1, 16)]
    assert summary["decision_snapshot_status_counts"] == {
        "failed": 0,
        "pending": 0,
        "succeeded": 15,
    }

    assert backend.requests
    for request in backend.requests:
        assert set(request) == {"messages", "response_format"}
        response_format = request["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        enum = response_format["json_schema"]["schema"]["properties"]["choice_id"][
            "enum"
        ]
        assert enum[0] == "ignore"

    teacher_rows = [
        json.loads(line)
        for line in (output / "groq_teacher_trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(teacher_rows) == 15
    assert all(row["label_source"] == "teacher_synthetic" for row in teacher_rows)
    assert all(
        row["label"]["reason"] == "synthetic audit rationale" for row in teacher_rows
    )
    assert all(row["provenance"] == "decision_snapshot" for row in teacher_rows)
    assert all(row["action_label_eligible"] is True for row in teacher_rows)
    assert all(row["rationale_training_eligible"] is True for row in teacher_rows)
    with sqlite3.connect(output / "real_llm_validation.db") as connection:
        connection.row_factory = sqlite3.Row
        snapshots = connection.execute(
            "SELECT * FROM diffusionguard_decision_snapshot ORDER BY decision_sequence"
        ).fetchall()
        assert len(snapshots) == 15
        for teacher, snapshot in zip(teacher_rows, snapshots, strict=True):
            assert teacher["sample_id"] == snapshot["decision_id"]
            assert json.loads(teacher["feed_post"]) == json.loads(snapshot["feed_json"])
            assert teacher["visible_post_ids"] == json.loads(
                snapshot["visible_post_ids_json"]
            )
            assert teacher["legal_choice_ids"] == json.loads(
                snapshot["legal_choice_ids_json"]
            )
            assert teacher["action_trace_rowid"] == snapshot["action_trace_rowid"]
            assert all(
                int(token.split(":", 1)[0].removeprefix("trace-"))
                < teacher["action_trace_rowid"]
                for token in teacher["behavior_history"]
            )
            prompt = next(
                request["messages"]
                for request in backend.requests
                if json.loads(request["messages"][1]["content"])["visible_feed"]
                == json.loads(snapshot["feed_json"])["posts"]
            )
            assert "selected_choice_id" not in json.dumps(prompt)
    secret = settings.groq_api_key.encode()
    for path in output.rglob("*"):
        if path.is_file():
            assert secret not in path.read_bytes()


async def run_retry_validation(
    tmp_path: Path,
    backend: StrictStructuredFakeBackend,
    output_name: str,
) -> tuple[dict[str, Any], Path]:
    settings = LLMSettings(
        enabled=True,
        provider="groq",
        llm_model="fake-structured-model",
        groq_api_key="unit-test-only-secret",
        temperature=0.0,
        max_tokens=512,
        max_retries=0,
        max_concurrency=1,
        max_calls_per_run=30,
        request_interval_seconds=0.0,
        cache_enabled=False,
        cache_path=tmp_path / f"{output_name}-unused-cache.sqlite3",
    )
    runtime = LLMRuntime(
        settings,
        cache=None,
        model_config=backend.model_config_dict,
        sleeper=lambda _: None,
    )
    output = tmp_path / output_name
    summary = await run_validation(
        settings,
        output,
        model_override=ManagedModelBackend(backend, runtime),
    )
    return summary, output


@pytest.mark.asyncio
async def test_real_oasis_five_by_three_recovers_selected_schema_failures(
    tmp_path: Path,
) -> None:
    baseline_summary, baseline_output = await run_retry_validation(
        tmp_path,
        StrictStructuredFakeBackend(),
        "fake-structured-no-retry-baseline",
    )
    failed_first = {2, 7, 11}
    backend = CorrectingStructuredFakeBackend(fail_first_decisions=failed_first)
    summary, output = await run_retry_validation(
        tmp_path, backend, "fake-structured-retry-success"
    )

    assert summary["status"] == "success"
    assert summary["completed_decisions"] == 15
    assert summary["teacher_examples"] == 15
    assert summary["physical_remote_attempts"] == 15 + len(failed_first)
    assert summary["first_attempt_count"] == 15
    assert summary["first_attempt_success_count"] == 12
    assert summary["first_attempt_success_rate"] == pytest.approx(12 / 15)
    assert summary["json_validate_failed_count"] == 3
    assert summary["structured_retry_attempt_count"] == 3
    assert summary["structured_retry_success_count"] == 3
    assert summary["structured_retry_failure_count"] == 0
    assert summary["post_retry_completed_decisions"] == 15
    assert summary["post_retry_success_rate"] == 1.0
    assert summary["unrecovered_error_count"] == 0
    assert summary["retry_count"] == 3
    assert len(backend.requests) == 18
    assert baseline_summary["logical_decisions"] == 15
    assert baseline_summary["physical_remote_attempts"] == 15
    assert summary["impression_count"] == baseline_summary["impression_count"]
    assert (
        sum(
            request["messages"][-1].get("content") == CORRECTION_PROMPT
            for request in backend.requests
        )
        == 3
    )

    with sqlite3.connect(output / "real_llm_validation.db") as connection:
        rows = connection.execute(
            "SELECT rowid FROM trace WHERE action IN (?, ?, ?, ?)",
            ("repost", "quote_post", "report_post", "do_nothing"),
        ).fetchall()
        refresh_trace_count = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE action = 'refresh'"
        ).fetchone()[0]
    with sqlite3.connect(baseline_output / "real_llm_validation.db") as connection:
        baseline_refresh_trace_count = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE action = 'refresh'"
        ).fetchone()[0]
    assert len(rows) == 15
    assert len({row[0] for row in rows}) == 15
    assert refresh_trace_count == baseline_refresh_trace_count == 15
    teacher_rows = [
        json.loads(line)
        for line in (output / "groq_teacher_trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(teacher_rows) == 15
    assert len({row["sample_id"] for row in teacher_rows}) == 15
    assert summary["impression_count"] == 30
    with sqlite3.connect(output / "real_llm_validation.db") as connection:
        snapshot_counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM diffusionguard_decision_snapshot "
                "GROUP BY status"
            )
        )
        snapshot_total = connection.execute(
            "SELECT COUNT(*) FROM diffusionguard_decision_snapshot"
        ).fetchone()[0]
    assert snapshot_total == 15
    assert snapshot_counts == {"succeeded": 15}


@pytest.mark.asyncio
async def test_second_schema_failure_never_enters_trace_cache_or_teacher_data(
    tmp_path: Path,
) -> None:
    backend = CorrectingStructuredFakeBackend(
        fail_first_decisions={3},
        fail_retry_decisions={3},
    )
    summary, output = await run_retry_validation(
        tmp_path, backend, "fake-structured-retry-failure"
    )

    assert summary["status"] == "degraded"
    assert summary["completed_decisions"] == 14
    assert summary["teacher_examples"] == 14
    assert summary["physical_remote_attempts"] == 16
    assert summary["structured_retry_attempt_count"] == 1
    assert summary["structured_retry_success_count"] == 0
    assert summary["structured_retry_failure_count"] == 1
    assert summary["unrecovered_error_count"] == 1
    assert summary["unrecovered_http_statuses"] == {"400": 1}
    assert "failed_generation" not in json.dumps(summary)
    assert not (
        tmp_path / "fake-structured-retry-failure-unused-cache.sqlite3"
    ).exists()

    with sqlite3.connect(output / "real_llm_validation.db") as connection:
        trace_count = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE action IN (?, ?, ?, ?)",
            ("repost", "quote_post", "report_post", "do_nothing"),
        ).fetchone()[0]
        refresh_trace_count = connection.execute(
            "SELECT COUNT(*) FROM trace WHERE action = 'refresh'"
        ).fetchone()[0]
    teacher_count = len(
        (output / "groq_teacher_trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert trace_count == teacher_count == 14
    assert refresh_trace_count == 15
    assert summary["impression_count"] == 30
    with sqlite3.connect(output / "real_llm_validation.db") as connection:
        snapshot_counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM diffusionguard_decision_snapshot "
                "GROUP BY status"
            )
        )
    assert snapshot_counts == {"failed": 1, "succeeded": 14}
