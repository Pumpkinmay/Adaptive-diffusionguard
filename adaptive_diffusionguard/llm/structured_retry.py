"""One bounded correction retry for Groq strict structured outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import (
    CallLimitExceeded,
    ManagedModelBackend,
    is_json_validate_failed,
)
from .structured_actions import DecisionSnapshot

CORRECTION_PROMPT = (
    "Previous response did not satisfy the allowed choice_id enum. "
    "Select exactly one choice_id from the current schema."
)


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """One schema-bound request and its caller-owned action-mask context."""

    messages: list[dict[str, Any]]
    response_format: dict[str, Any]
    context: Any


@dataclass(frozen=True, slots=True)
class StructuredAttemptResult:
    response: Any
    request: StructuredRequest
    retried: bool


def _request_from_snapshot(
    snapshot: DecisionSnapshot, *, correction: bool
) -> StructuredRequest:
    messages = snapshot.messages_value()
    if correction:
        messages.append({"role": "user", "content": CORRECTION_PROMPT})
    return StructuredRequest(
        messages=messages,
        response_format=snapshot.response_format_value(),
        context=snapshot,
    )


async def request_with_structured_correction(
    model: ManagedModelBackend,
    snapshot: DecisionSnapshot,
) -> StructuredAttemptResult:
    """Issue one first attempt and at most one narrowly qualified retry.

    Provider error payloads are never added to messages. Both attempts use the
    same immutable feed, legal choices, and response schema captured before
    the first request; the retry differs only by its fixed correction prompt.
    """
    runtime = model.runtime
    initial = _request_from_snapshot(snapshot, correction=False)
    runtime.begin_first_attempt()
    try:
        response = await model.structured_arun(
            initial.messages,
            response_format=initial.response_format,
            defer_unrecovered_error=True,
        )
        return StructuredAttemptResult(response, initial, retried=False)
    except Exception as first_error:
        if not is_json_validate_failed(first_error):
            if not isinstance(first_error, CallLimitExceeded):
                runtime.record_final_error(first_error)
            raise
        runtime.record_json_validate_failed()

        if runtime.remaining_calls <= 0:
            runtime.record_final_error(first_error)
            raise

        retry_request = _request_from_snapshot(snapshot, correction=True)

        try:
            runtime.begin_structured_retry()
        except CallLimitExceeded:
            runtime.record_final_error(first_error)
            raise

        try:
            response = await model.structured_arun(
                retry_request.messages,
                response_format=retry_request.response_format,
                cache_read=False,
                defer_unrecovered_error=True,
            )
            return StructuredAttemptResult(response, retry_request, retried=True)
        except Exception as retry_error:
            if is_json_validate_failed(retry_error):
                runtime.record_json_validate_failed()
            runtime.record_structured_retry_failure()
            if isinstance(retry_error, CallLimitExceeded):
                runtime.record_final_error(first_error)
            else:
                runtime.record_final_error(retry_error)
            raise
