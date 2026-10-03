"""Strict action normalization for teacher and OASIS adapter outputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ActionName = Literal["repost", "quote", "report", "ignore"]
VALID_ACTIONS = frozenset({"repost", "quote", "report", "ignore"})


class ActionOutput(BaseModel):
    """Only action object allowed to enter behavior simulation or training."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: ActionName
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @classmethod
    def parse_json(cls, value: str) -> ActionOutput:
        return cls.model_validate_json(value)

    def to_json(self) -> str:
        return self.model_dump_json()


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    output: ActionOutput
    json_valid: bool
    action_valid: bool
    repair_attempted: bool
    failure_reason: str | None = None


@dataclass(slots=True)
class ActionValidationMetrics:
    total: int = 0
    json_valid: int = 0
    action_valid: int = 0
    repair_attempts: int = 0

    def observe(self, result: NormalizationResult) -> None:
        self.total += 1
        self.json_valid += int(result.json_valid)
        self.action_valid += int(result.action_valid)
        self.repair_attempts += int(result.repair_attempted)

    def to_dict(self) -> dict[str, float | int]:
        denominator = max(1, self.total)
        return {
            "total": self.total,
            "json_valid_rate": self.json_valid / denominator,
            "action_valid_rate": self.action_valid / denominator,
            "repair_rate": self.repair_attempts / denominator,
        }


def _failure_kind(raw: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, "invalid_json"
    if not isinstance(parsed, dict) or parsed.get("action") not in VALID_ACTIONS:
        return True, "invalid_action"
    return True, "invalid_schema"


def normalize_action(
    raw: str,
    *,
    repair: Callable[[str], str] | None = None,
) -> NormalizationResult:
    """Parse once, optionally repair once, then safely fall back to ignore."""
    initial_json_valid, initial_failure = _failure_kind(raw)
    try:
        output = ActionOutput.model_validate_json(raw)
        return NormalizationResult(output, True, True, False)
    except (ValidationError, ValueError, TypeError):
        pass

    if repair is not None:
        repaired_raw = repair(raw)
        repaired_json_valid, repaired_failure = _failure_kind(repaired_raw)
        try:
            output = ActionOutput.model_validate_json(repaired_raw)
            return NormalizationResult(output, True, True, True)
        except (ValidationError, ValueError, TypeError):
            failure = repaired_failure
            json_valid = repaired_json_valid
    else:
        failure = initial_failure
        json_valid = initial_json_valid

    fallback = ActionOutput(
        action="ignore",
        confidence=0.0,
        reason=f"safe fallback after {failure}",
    )
    return NormalizationResult(
        fallback,
        json_valid=json_valid,
        action_valid=False,
        repair_attempted=repair is not None,
        failure_reason=failure,
    )


def extract_response_text(response: Any) -> str:
    """Extract assistant content without altering native tool-call responses."""
    choices = getattr(response, "choices", None)
    if choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if isinstance(content, str):
            return content
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, str):
                return content
    if isinstance(response, str):
        return response
    raise ValueError("model response contains no textual assistant content")
