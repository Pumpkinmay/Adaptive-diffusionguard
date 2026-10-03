"""COSREF + LLM joint experiment v2.1 semantic-consistency protocol.

This module is version-isolated from v1 and v2.  It keeps the v2 ordered legal
action list, but requires the model to repeat the selected action and direct
post ID so that an index/meaning mismatch is rejected before dispatch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationError,
)

from adaptive_diffusionguard.llm.action_gateway import (
    ActionDecisionError,
    ActionDecisionGateway,
)
from adaptive_diffusionguard.llm.actions import extract_response_text
from adaptive_diffusionguard.llm.model_factory import LLMSettings, create_llm_model
from adaptive_diffusionguard.llm.runtime import (
    CallLimitExceeded,
    LLMRuntime,
    ManagedModelBackend,
    RedactedProviderError,
    ResponseCache,
    is_json_validate_failed,
)
from adaptive_diffusionguard.llm.structured_actions import ActionMask, DecisionSnapshot
from adaptive_diffusionguard.storage.decision_snapshots import DecisionSnapshotStore

from . import cosref_llm_joint as v1
from . import cosref_llm_joint_v2 as v2

BackendName = Literal["fake", "groq"]
ActionType = Literal["ignore", "report", "repost", "quote"]
PROTOCOL_VERSION = "fixed-choice-semantic-v2.1"
SEMANTIC_CORRECTION_PROMPT = (
    "choice_index、action_type和target_post_id必须共同描述同一个列出的合法选项"
)
NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]


class SemanticActionResponse(BaseModel):
    """Fixed provider response; legal values remain local snapshot state."""

    model_config = ConfigDict(extra="forbid", strict=True)

    choice_index: StrictInt
    action_type: ActionType
    target_post_id: StrictInt
    quote_text: StrictStr
    rationale: NonEmptyText


class LocalSchemaFailure(ValueError):
    """A provider response failed the fixed local Pydantic schema."""


class InvalidChoiceIndex(ValueError):
    """A strict integer index is outside the frozen action list."""


class SemanticConsistencyError(ValueError):
    """Base class for redundant-field disagreement."""


class ChoiceActionMismatch(SemanticConsistencyError):
    """The repeated action type disagrees with the indexed action."""


class ChoiceTargetMismatch(SemanticConsistencyError):
    """The repeated direct post ID disagrees with the indexed action."""


class QuoteTextContractFailure(SemanticConsistencyError):
    """quote_text violates the fixed all-actions field contract."""


class CurrentStateChoiceInvalid(SemanticConsistencyError):
    """The frozen option is no longer legal immediately before dispatch."""


@dataclass(frozen=True, slots=True)
class SemanticAttemptResult:
    response: Any
    messages: list[dict[str, Any]]
    response_format: dict[str, Any]
    parsed: SemanticActionResponse
    choice_id: str
    retried: bool
    semantic_correction: bool


@dataclass(slots=True)
class SemanticAuditMetrics:
    """Attempt-level semantic diagnostics, separate from logical failures."""

    semantic_consistency_failure_count: int = 0
    choice_action_mismatch_count: int = 0
    choice_target_mismatch_count: int = 0
    quote_text_contract_failure_count: int = 0
    semantic_correction_attempt_count: int = 0
    semantic_correction_success_count: int = 0
    semantic_correction_failure_count: int = 0

    def record_mismatch(self, exc: BaseException) -> None:
        self.semantic_consistency_failure_count += 1
        if isinstance(exc, ChoiceActionMismatch):
            self.choice_action_mismatch_count += 1
        elif isinstance(exc, ChoiceTargetMismatch):
            self.choice_target_mismatch_count += 1
        elif isinstance(exc, QuoteTextContractFailure):
            self.quote_text_contract_failure_count += 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class FailureAccounting:
    """Mutually exclusive final logical-decision categories."""

    provider_failure_count: int = 0
    provider_json_validate_failed_count: int = 0
    local_schema_failure_count: int = 0
    invalid_choice_index_count: int = 0
    semantic_final_failure_count: int = 0
    dispatcher_failure_count: int = 0
    completed_decision_count: int = 0
    incomplete_decision_count: int = 0
    unrecovered_logical_error_count: int = 0

    def record_completed(self) -> None:
        self.completed_decision_count += 1

    def record_failure(self, category: str) -> None:
        if category == "provider_json_validate_failed":
            self.provider_failure_count += 1
            self.provider_json_validate_failed_count += 1
        elif category == "provider_failure":
            self.provider_failure_count += 1
        elif category == "local_schema_failure":
            self.local_schema_failure_count += 1
        elif category == "invalid_choice_index":
            self.invalid_choice_index_count += 1
        elif category == "semantic_consistency_failure":
            self.semantic_final_failure_count += 1
        elif category == "dispatcher_failure":
            self.dispatcher_failure_count += 1
        else:
            raise ValueError(f"unsupported failure category: {category}")
        self.incomplete_decision_count += 1
        self.unrecovered_logical_error_count += 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SemanticActionMaskBuilder(v2.RootAwareActionMaskBuilder):
    """V2 root-aware legality with a v2.1 state fingerprint."""

    def state_identifier(
        self, user_id: int, visible_post_ids: set[int] | frozenset[int]
    ) -> str:
        posts = []
        for post_id in sorted(visible_post_ids):
            row = self.connection.execute(
                "SELECT user_id, original_post_id, quote_content FROM post "
                "WHERE post_id = ?",
                (post_id,),
            ).fetchone()
            if row is not None:
                posts.append(
                    [post_id, int(row[0]), row[1], row[2], self.root_post_id(post_id)]
                )
        material = {
            "protocol_version": PROTOCOL_VERSION,
            "user_id": user_id,
            "posts": posts,
            "reported_roots": sorted(self._reported_roots(user_id)),
            "reposted_roots": sorted(self._reposted_roots(user_id)),
        }
        return hashlib.sha256(v1._canonical(material).encode()).hexdigest()


def semantic_response_format() -> dict[str, Any]:
    """Constant strict schema: no dynamic choice enum and no tool calling."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "adaptive_diffusionguard_action_semantic_v2_1",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice_index": {"type": "integer"},
                    "action_type": {
                        "type": "string",
                        "enum": ["ignore", "report", "repost", "quote"],
                    },
                    "target_post_id": {"type": "integer"},
                    "quote_text": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "choice_index",
                    "action_type",
                    "target_post_id",
                    "quote_text",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    }


def _choice_record(
    index: int,
    choice_id: str,
    builder: SemanticActionMaskBuilder,
) -> dict[str, Any]:
    action, separator, raw_post = choice_id.partition(":")
    if not separator:
        return {
            "choice_index": index,
            "action_type": "ignore",
            "action": "ignore",
            "target_post_id": 0,
            "selected_post_id": None,
            "canonical_root_post_id": None,
            "description": "ignore without a platform mutation",
        }
    post_id = int(raw_post)
    root = builder.root_post_id(post_id)
    return {
        "choice_index": index,
        "action_type": action,
        "action": action,
        "target_post_id": post_id,
        "selected_post_id": post_id,
        "canonical_root_post_id": root,
        "description": f"{action} visible post {post_id} (canonical root {root})",
    }


def build_semantic_messages(
    *,
    profile: str,
    feed: dict[str, Any],
    mask: ActionMask,
    builder: SemanticActionMaskBuilder,
) -> list[dict[str, str]]:
    system = (
        "You are a synthetic ordinary-user behavior simulator, not a safety "
        "classifier. Follow the supplied synthetic profile. Select exactly one "
        "ordered legal action. choice_index, action_type, and target_post_id must "
        "all describe that same listed option. Use target_post_id=0 for ignore. "
        "Use non-empty quote_text only for quote; otherwise use an empty string. "
        "Hidden risk labels, treatment identity, and network-control parameters "
        "are not supplied. These outputs are not observed human behavior or "
        "training labels."
    )
    ordered = [
        _choice_record(index, choice, builder)
        for index, choice in enumerate(mask.choices)
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "agent_role": "behavior_simulator",
        "profile": profile,
        "visible_feed": feed.get("posts", []),
        "ordered_legal_actions": ordered,
        "instruction": (
            "Return fixed-schema fields for exactly one listed action; do not "
            "invent or renumber an option."
        ),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def parse_semantic_response(response: Any) -> SemanticActionResponse:
    choices = response.get("choices") if isinstance(response, dict) else getattr(
        response, "choices", None
    )
    if not choices:
        raise LocalSchemaFailure("semantic response contains no choices")
    choice = choices[0]
    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    if finish_reason == "length":
        raise LocalSchemaFailure("semantic response was truncated")
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
        raise LocalSchemaFailure("v2.1 response unexpectedly contains tool calls")
    try:
        return SemanticActionResponse.model_validate_json(
            extract_response_text(response)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LocalSchemaFailure("semantic response failed local schema") from exc


def _expected_semantics(choice_id: str) -> tuple[str, int]:
    action, separator, raw_post = choice_id.partition(":")
    return (action, int(raw_post)) if separator else ("ignore", 0)


def validate_semantic_binding(
    snapshot: DecisionSnapshot,
    parsed: SemanticActionResponse,
) -> str:
    if (
        isinstance(parsed.choice_index, bool)
        or parsed.choice_index < 0
        or parsed.choice_index >= len(snapshot.legal_choice_ids)
    ):
        raise InvalidChoiceIndex("choice_index is outside the frozen action list")
    choice_id = snapshot.legal_choice_ids[parsed.choice_index]
    expected_action, expected_target = _expected_semantics(choice_id)
    if parsed.action_type != expected_action:
        raise ChoiceActionMismatch(
            "action_type does not match the indexed legal option"
        )
    if parsed.target_post_id != expected_target:
        raise ChoiceTargetMismatch(
            "target_post_id does not match the indexed legal option"
        )
    if parsed.action_type == "quote" and not parsed.quote_text.strip():
        raise QuoteTextContractFailure("quote requires non-empty quote_text")
    if parsed.action_type != "quote" and parsed.quote_text != "":
        raise QuoteTextContractFailure(
            "non-quote actions require an empty quote_text"
        )
    return choice_id


def revalidate_semantic_choice(
    snapshot: DecisionSnapshot,
    parsed: SemanticActionResponse,
    builder: SemanticActionMaskBuilder,
) -> tuple[str, ActionMask]:
    choice_id = validate_semantic_binding(snapshot, parsed)
    current = builder.build(snapshot.user_id, frozenset(snapshot.visible_post_ids))
    if choice_id not in current.choices:
        raise CurrentStateChoiceInvalid(
            "frozen semantically bound choice is no longer legal"
        )
    # Root resolution is also a final existence/cycle check.
    if parsed.target_post_id:
        builder.root_post_id(parsed.target_post_id)
    return choice_id, current


async def request_semantic_with_correction(
    model: Any,
    snapshot: DecisionSnapshot,
    metrics: SemanticAuditMetrics,
) -> SemanticAttemptResult:
    """Issue at most one correction using exactly the frozen snapshot."""
    runtime = model.runtime
    response_format = snapshot.response_format_value()
    base_messages = snapshot.messages_value()

    async def one(
        messages: list[dict[str, Any]], *, cache_read: bool
    ) -> tuple[Any, SemanticActionResponse, str]:
        response = await model.structured_arun(
            messages,
            response_format=response_format,
            cache_read=cache_read,
            defer_unrecovered_error=True,
        )
        parsed = parse_semantic_response(response)
        return response, parsed, validate_semantic_binding(snapshot, parsed)

    runtime.begin_first_attempt()
    first_semantic = False
    try:
        response, parsed, choice_id = await one(base_messages, cache_read=True)
        return SemanticAttemptResult(
            response,
            base_messages,
            response_format,
            parsed,
            choice_id,
            False,
            False,
        )
    except Exception as first_error:
        first_semantic = isinstance(first_error, SemanticConsistencyError)
        if first_semantic:
            metrics.record_mismatch(first_error)
            metrics.semantic_correction_attempt_count += 1
        retryable = first_semantic or isinstance(
            first_error, (LocalSchemaFailure, InvalidChoiceIndex)
        ) or is_json_validate_failed(first_error)
        if not retryable:
            if not isinstance(first_error, CallLimitExceeded):
                runtime.record_final_error(first_error)
            raise
        if is_json_validate_failed(first_error):
            runtime.record_json_validate_failed()
        if runtime.remaining_calls <= 0:
            if first_semantic:
                metrics.semantic_correction_failure_count += 1
            elif not isinstance(first_error, (LocalSchemaFailure, InvalidChoiceIndex)):
                runtime.record_final_error(first_error)
            raise
        retry_messages = snapshot.messages_value()
        retry_messages.append(
            {"role": "user", "content": SEMANTIC_CORRECTION_PROMPT}
        )
        runtime.begin_structured_retry()
        try:
            response, parsed, choice_id = await one(
                retry_messages, cache_read=False
            )
            return SemanticAttemptResult(
                response,
                retry_messages,
                response_format,
                parsed,
                choice_id,
                True,
                first_semantic,
            )
        except Exception as retry_error:
            if isinstance(retry_error, SemanticConsistencyError):
                metrics.record_mismatch(retry_error)
            if first_semantic:
                metrics.semantic_correction_failure_count += 1
            if is_json_validate_failed(retry_error):
                runtime.record_json_validate_failed()
                runtime.record_final_error(retry_error)
            elif not isinstance(
                retry_error,
                (LocalSchemaFailure, InvalidChoiceIndex, SemanticConsistencyError),
            ) and not isinstance(retry_error, CallLimitExceeded):
                runtime.record_final_error(retry_error)
            runtime.record_structured_retry_failure()
            raise


class FakeSemanticModel:
    """Deterministic v2.1 backend with auditable failure injection."""

    def __init__(
        self,
        cache_path: Path,
        *,
        max_attempts: int = 20,
        failure_mode: str = "once_semantic_mismatch",
    ) -> None:
        settings = LLMSettings(
            enabled=False,
            provider="groq",
            llm_model="fake-structured-joint-semantic-v2-1",
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=60.0,
            max_retries=0,
            max_concurrency=1,
            max_calls_per_run=max_attempts,
            cache_enabled=True,
            request_interval_seconds=0.0,
            cache_path=cache_path,
        )
        self.runtime = LLMRuntime(
            settings,
            cache=ResponseCache(cache_path),
            model_config={
                "temperature": 0.0,
                "max_tokens": 512,
                "protocol_version": PROTOCOL_VERSION,
            },
            sleeper=lambda _: None,
        )
        self.max_attempts = max_attempts
        self.failure_mode = failure_mode
        self.local_attempts = 0
        self.successful_responses = 0

    @staticmethod
    def _normal_value(messages: list[dict[str, Any]]) -> dict[str, Any]:
        index, rationale = v2.FakeIndexModel._select(messages)
        payload = json.loads(str(messages[1]["content"]))
        option = payload["ordered_legal_actions"][index]
        action = str(option["action_type"])
        return {
            "choice_index": index,
            "action_type": action,
            "target_post_id": int(option["target_post_id"]),
            "quote_text": (
                "Synthetic corrective context for the selected visible claim."
                if action == "quote"
                else ""
            ),
            "rationale": rationale,
        }

    @staticmethod
    def _first_option(
        messages: list[dict[str, Any]], action: str, *, negate: bool = False
    ) -> dict[str, Any]:
        payload = json.loads(str(messages[1]["content"]))
        for option in payload["ordered_legal_actions"]:
            matches = str(option["action_type"]) == action
            if matches != negate:
                return option
        raise AssertionError("required fake action option is absent")

    async def structured_arun(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> dict[str, Any]:
        del cache_read, defer_unrecovered_error
        if response_format != semantic_response_format():
            raise AssertionError("fake v2.1 received a non-semantic schema")
        if self.local_attempts >= self.max_attempts:
            raise CallLimitExceeded("fake backend physical-attempt cap reached")
        self.local_attempts += 1
        correction = bool(
            messages and messages[-1].get("content") == SEMANTIC_CORRECTION_PROMPT
        )
        mode = self.failure_mode
        if mode == "once_json_validate" and not correction:
            raise RedactedProviderError(
                "fake strict validation rejection", 400, "json_validate_failed"
            )
        value = self._normal_value(messages)
        inject_once = mode == "once_semantic_mismatch" and not correction
        if inject_once or mode in {"always_semantic_mismatch", "target_mismatch"}:
            value["target_post_id"] = int(value["target_post_id"]) + 10_000
        elif mode == "action_mismatch":
            value["action_type"] = (
                "report" if value["action_type"] != "report" else "repost"
            )
        elif mode == "ignore_target_nonzero":
            option = self._first_option(messages, "ignore")
            value.update(
                choice_index=int(option["choice_index"]),
                action_type="ignore",
                target_post_id=1,
                quote_text="",
            )
        elif mode == "action_target_zero":
            option = self._first_option(messages, "ignore", negate=True)
            value.update(
                choice_index=int(option["choice_index"]),
                action_type=str(option["action_type"]),
                target_post_id=0,
                quote_text=(
                    "context" if option["action_type"] == "quote" else ""
                ),
            )
        elif mode == "quote_empty":
            option = self._first_option(messages, "quote")
            value.update(
                choice_index=int(option["choice_index"]),
                action_type="quote",
                target_post_id=int(option["target_post_id"]),
                quote_text="",
            )
        elif mode == "nonquote_text":
            option = self._first_option(messages, "quote", negate=True)
            value.update(
                choice_index=int(option["choice_index"]),
                action_type=str(option["action_type"]),
                target_post_id=int(option["target_post_id"]),
                quote_text="unexpected text",
            )
        elif mode == "non_integer":
            value["choice_index"] = str(value["choice_index"])
        self.successful_responses += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            value, ensure_ascii=False, sort_keys=True
                        )
                    },
                }
            ]
        }

    def cache_structured_response(
        self,
        response: Any,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
    ) -> bool:
        return self.runtime.store_response(
            response,
            messages=messages,
            response_format=response_format,
            model_config={
                "temperature": 0.0,
                "max_tokens": 512,
                "protocol_version": PROTOCOL_VERSION,
            },
        )

    def discard_structured_response(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
    ) -> None:
        self.runtime.discard_response(
            messages=messages,
            response_format=response_format,
            model_config={
                "temperature": 0.0,
                "max_tokens": 512,
                "protocol_version": PROTOCOL_VERSION,
            },
        )


def _load_config(path: Path) -> dict[str, Any]:
    config = v1._load_config(path)
    if config.get("version") != "2.1":
        raise v1.JointConfigurationError("v2.1 config version must equal '2.1'")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise v1.JointConfigurationError("v2.1 semantic protocol is required")
    if config.get("agent_role") != "behavior_simulator":
        raise v1.JointConfigurationError(
            "v2.1 requires agent_role=behavior_simulator"
        )
    selections = list(
        config.get("semantic_micro_pilot", {}).get("selections", [])
    )
    maximum = int(
        config.get("semantic_micro_pilot", {}).get(
            "maximum_physical_requests", 0
        )
    )
    if len(selections) != 12 or maximum != 24:
        raise v1.JointConfigurationError(
            "semantic micro pilot must contain 12 decisions and 24 requests"
        )
    keys = {
        (
            str(row["condition_id"]),
            str(row["strategy"]),
            int(row["timestep"]),
            int(row["user_id"]),
        )
        for row in selections
    }
    if len(keys) != 12:
        raise v1.JointConfigurationError(
            "semantic micro-pilot selections must be unique"
        )
    if {key[0] for key in keys} != {
        "strong-community",
        "moderate-mixing",
        "weak-community",
    }:
        raise v1.JointConfigurationError(
            "semantic micro pilot must cover all network conditions"
        )
    if {key[1] for key in keys} != set(v1.STRATEGIES):
        raise v1.JointConfigurationError(
            "semantic micro pilot must cover all exposure strategies"
        )
    profiles = {
        int(agent["user_id"]): str(agent["profile_id"])
        for agent in config["agents"]
    }
    if len({profiles[key[3]] for key in keys}) < 4:
        raise v1.JointConfigurationError(
            "semantic micro pilot must cover at least four behavior profiles"
        )
    return config


def _selection_map(
    config: Mapping[str, Any], *, semantic_micro_pilot: bool
) -> dict[tuple[str, str], set[tuple[int, int]]] | None:
    if not semantic_micro_pilot:
        return None
    result: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for row in config["semantic_micro_pilot"]["selections"]:
        result[(str(row["condition_id"]), str(row["strategy"]))].add(
            (int(row["timestep"]), int(row["user_id"]))
        )
    return dict(result)


def build_plan(
    config_path: Path,
    backend: BackendName,
    output: Path,
    *,
    semantic_micro_pilot: bool,
) -> dict[str, Any]:
    config = _load_config(config_path)
    networks = [
        v1._generate_network(config, condition)
        for condition in config["network"]["conditions"]
    ]
    selected = _selection_map(
        config, semantic_micro_pilot=semantic_micro_pilot
    )
    units = []
    for network in networks:
        for strategy in v1.STRATEGIES:
            logical = (
                len(selected.get((network.condition_id, strategy), set()))
                if selected is not None
                else 10
            )
            if logical:
                units.append(
                    {
                        "condition_id": network.condition_id,
                        "strategy": strategy,
                        "unit_id": f"{network.condition_id}--{strategy}",
                        "measured_mu": network.measured_mu,
                        "logical_decisions": logical,
                        "maximum_physical_requests": logical * 2,
                    }
                )
    logical = sum(int(unit["logical_decisions"]) for unit in units)
    maximum = sum(int(unit["maximum_physical_requests"]) for unit in units)
    return {
        "agent_role": "behavior_simulator",
        "backend": backend,
        "config_sha256": v1._sha256(config_path),
        "experiment_id": config["experiment_id"],
        "experiment_units": len(units),
        "logical_decisions": logical,
        "maximum_physical_requests_total": maximum,
        "semantic_micro_pilot": bool(semantic_micro_pilot),
        "model_created": False,
        "oracle_synthetic_risk_labels": True,
        "output_created": False,
        "output_path": str(output),
        "protocol_version": PROTOCOL_VERSION,
        "request_interval_seconds": float(
            config["llm"]["request_interval_seconds"]
        ),
        "response_schema_dynamic": False,
        "response_fields": [
            "choice_index",
            "action_type",
            "target_post_id",
            "quote_text",
            "rationale",
        ],
        "tools_sent": False,
        "tool_choice_sent": False,
        "temperature": 0.0,
        "max_tokens": 512,
        "max_concurrency": 1,
        "units": units,
    }


def _model_for_unit(
    backend: BackendName,
    config: Mapping[str, Any],
    cache_path: Path,
    logical_decisions: int,
    llm_settings: LLMSettings | None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None],
) -> FakeSemanticModel | ManagedModelBackend:
    limit = logical_decisions * 2
    if backend == "fake":
        return FakeSemanticModel(
            cache_path,
            max_attempts=limit,
            failure_mode=str(
                config.get("fake_backend", {}).get(
                    "failure_mode", "once_semantic_mismatch"
                )
            ),
        )
    if llm_settings is None:
        raise RuntimeError("groq backend requires explicit LLM settings")
    bounded = replace(
        llm_settings,
        enabled=True,
        provider="groq",
        temperature=0.0,
        max_tokens=512,
        max_concurrency=1,
        max_calls_per_run=limit,
        request_interval_seconds=max(
            float(config["llm"]["request_interval_seconds"]),
            llm_settings.request_interval_seconds,
        ),
        cache_enabled=True,
        cache_path=cache_path,
    )
    model = model_factory(bounded)
    if model is None:
        raise RuntimeError("groq model creation is disabled")
    return model


def _failure_category(exc: BaseException) -> str:
    if is_json_validate_failed(exc):
        return "provider_json_validate_failed"
    if isinstance(exc, (RedactedProviderError, CallLimitExceeded)):
        return "provider_failure"
    if isinstance(exc, LocalSchemaFailure):
        return "local_schema_failure"
    if isinstance(exc, InvalidChoiceIndex):
        return "invalid_choice_index"
    if isinstance(exc, SemanticConsistencyError):
        return "semantic_consistency_failure"
    return "dispatcher_failure"


async def _run_unit(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    network: v1.NetworkSpec,
    strategy: v1.StrategyName,
    backend: BackendName,
    directory: Path,
    selected: set[tuple[int, int]] | None,
    llm_settings: LLMSettings | None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, directory / "preregistered_config.json")
    v1._write_json(directory / "network.json", network.public_dict())
    platform, logical_posts, root_risk, initialization = await v1._initialize_platform(
        config, network, strategy, directory / "experiment.db"
    )
    platform.run_id = (
        f"joint-v2-1-{network.condition_id}-{strategy}-{network.seed}"
    )
    initialization["run_id"] = platform.run_id
    initialization["protocol_version"] = PROTOCOL_VERSION
    initialization["agent_role"] = "behavior_simulator"
    v1._write_json(
        directory / "initialization.json",
        {
            **initialization,
            "logical_posts": logical_posts,
            "root_risk_scores": root_risk,
        },
    )
    v1._write_json(directory / "exposure_policy.json", initialization["policy"])
    expected = len(selected) if selected is not None else 10
    model = _model_for_unit(
        backend,
        config,
        directory / "response_cache_v2_1.sqlite3",
        expected,
        llm_settings,
        model_factory,
    )
    store = DecisionSnapshotStore(
        platform.db, now_factory=lambda: v1.FIXED_SNAPSHOT_TIME
    )
    builder = SemanticActionMaskBuilder(platform.db)
    profiles = {int(row["user_id"]): row for row in config["agents"]}
    decisions: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    accounting = FailureAccounting()
    semantic_metrics = SemanticAuditMetrics()
    sequence = 0
    started = time.monotonic()
    try:
        for timestep, order in enumerate(
            config["simulation"]["active_user_order"], start=1
        ):
            platform.sandbox_clock.time_step = timestep
            for raw_user in order:
                user_id = int(raw_user)
                if selected is not None and (timestep, user_id) not in selected:
                    continue
                sequence += 1
                adapter = v1.PlatformActionAdapter(platform, user_id)
                gateway = ActionDecisionGateway(adapter)
                feed = await adapter.refresh()
                gateway.update_visible_feed(feed)
                mask = builder.build(user_id, gateway.visible_post_ids)
                response_format = semantic_response_format()
                profile = profiles[user_id]
                messages = build_semantic_messages(
                    profile=str(profile["description"]),
                    feed=feed,
                    mask=mask,
                    builder=builder,
                )
                isolated = v2._prompt_isolated(
                    messages, strategy, network.condition_id
                )
                snapshot = DecisionSnapshot.capture(
                    user_id=user_id,
                    timestep=timestep,
                    feed=feed,
                    visible_post_ids=gateway.visible_post_ids,
                    mask=mask,
                    response_format=response_format,
                    messages=messages,
                    state_identifier=builder.state_identifier(
                        user_id, gateway.visible_post_ids
                    ),
                )
                pending = store.create_pending(
                    snapshot,
                    run_id=platform.run_id,
                    agent_id=user_id,
                    decision_sequence=sequence,
                    user_profile=str(profile["description"]),
                    community=network.communities[user_id],
                    behavior_history=v1._history_before(platform, user_id),
                    platform_notice="Synthetic joint-experiment v2.1 behavior context.",
                )
                local_shadow, shadow_unchanged = v1._threshold_shadow(
                    platform=platform,
                    network=network,
                    config=config,
                    snapshot=snapshot,
                )
                for row in local_shadow:
                    row["decision_id"] = pending.decision_id
                shadow_rows.extend(local_shadow)
                request_messages = messages
                request_format = response_format
                result: SemanticAttemptResult | None = None
                completed = False
                try:
                    result = await request_semantic_with_correction(
                        model, snapshot, semantic_metrics
                    )
                    request_messages = result.messages
                    request_format = result.response_format
                    choice_id, current = revalidate_semantic_choice(
                        snapshot, result.parsed, builder
                    )
                    context = v1._post_context(platform, user_id, choice_id)
                    before = v1._latest_trace(platform)
                    dispatch_text = (
                        result.parsed.quote_text
                        if result.parsed.action_type == "quote"
                        else result.parsed.rationale
                    )
                    await gateway.dispatch_choice(
                        choice_id, dispatch_text, current.choices
                    )
                    trace_rowid = v1._trace_after(
                        platform, before, user_id, choice_id
                    )
                    if trace_rowid is None:
                        raise ActionDecisionError(
                            "dispatcher succeeded without a trace"
                        )
                    store.mark_succeeded(
                        pending.decision_id,
                        selected_choice_id=choice_id,
                        rationale=result.parsed.rationale,
                        action_trace_rowid=trace_rowid,
                    )
                    completed = True
                    accounting.record_completed()
                    if result.semantic_correction:
                        semantic_metrics.semantic_correction_success_count += 1
                    model.runtime.record_structured_completion(
                        retried=result.retried
                    )
                    model.cache_structured_response(
                        result.response,
                        messages=request_messages,
                        response_format=request_format,
                    )
                    action = result.parsed.action_type
                    decisions.append(
                        {
                            "action": action,
                            "action_trace_rowid": trace_rowid,
                            "agent_role": "behavior_simulator",
                            "backend": backend,
                            "canonical_root_post_id": context["root_post_id"],
                            "choice_index": result.parsed.choice_index,
                            "community_relation": context["community_relation"],
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "legal_choice_count": len(snapshot.legal_choice_ids),
                            "legal_choice_valid": True,
                            "model_prompt_isolated_from_treatment": isolated,
                            "profile_id": str(profile["profile_id"]),
                            "protocol_version": PROTOCOL_VERSION,
                            "quote_text_contract_valid": True,
                            "quote_text_sha256": hashlib.sha256(
                                result.parsed.quote_text.encode()
                            ).hexdigest(),
                            "rationale_fact_consistent": v2._rationale_fact_consistent(
                                action,
                                context["root_risk_score"],
                                result.parsed.rationale,
                            ),
                            "rationale_sha256": hashlib.sha256(
                                result.parsed.rationale.encode()
                            ).hexdigest(),
                            "rationale_training_eligible": False,
                            "root_post_id": context["root_post_id"],
                            "root_risk_score": context["root_risk_score"],
                            "run_id": platform.run_id,
                            "schema_valid": True,
                            "selected_choice_id": choice_id,
                            "selected_post_id": context["target_post_id"],
                            "semantic_consistent": True,
                            "semantic_correction_used": result.semantic_correction,
                            "status": "succeeded",
                            "strategy": strategy,
                            "structured_retry_used": result.retried,
                            "target_post_id": result.parsed.target_post_id,
                            "threshold_shadow_non_mutating": shadow_unchanged,
                            "timestep": timestep,
                            "training_eligible": False,
                            "user_id": user_id,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - experiment boundary
                    model.discard_structured_response(
                        messages=request_messages,
                        response_format=request_format,
                    )
                    category = _failure_category(exc)
                    accounting.record_failure(category)
                    if not completed:
                        status = platform.db.execute(
                            "SELECT status FROM diffusionguard_decision_snapshot "
                            "WHERE decision_id = ?",
                            (pending.decision_id,),
                        ).fetchone()
                        if status is not None and status[0] == "pending":
                            store.mark_failed(pending.decision_id, category)
                    decisions.append(
                        {
                            "agent_role": "behavior_simulator",
                            "backend": backend,
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "failure_category": category,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "model_prompt_isolated_from_treatment": isolated,
                            "profile_id": str(profile["profile_id"]),
                            "protocol_version": PROTOCOL_VERSION,
                            "rationale_training_eligible": False,
                            "schema_valid": False,
                            "semantic_consistent": False,
                            "status": "failed",
                            "strategy": strategy,
                            "threshold_shadow_non_mutating": shadow_unchanged,
                            "timestep": timestep,
                            "training_eligible": False,
                            "user_id": user_id,
                        }
                    )
        runtime = model.runtime.stats.to_dict()
        if isinstance(model, FakeSemanticModel):
            runtime.update(
                {
                    "physical_backend_attempts": model.local_attempts,
                    "physical_remote_attempts": 0,
                    "successful_backend_responses": model.successful_responses,
                    "successful_provider_responses": 0,
                }
            )
        runtime["protocol_version"] = PROTOCOL_VERSION
        runtime["failure_accounting"] = accounting.to_dict()
        runtime.update(semantic_metrics.to_dict())
        behavior = v1._unit_behavior_metrics(
            platform, decisions, shadow_rows, initialization["policy"]
        )
        counts = store.status_counts()
        summary = {
            "agent_role": "behavior_simulator",
            "backend": backend,
            "behavior": behavior,
            "completed_decisions": counts["succeeded"],
            "condition_id": network.condition_id,
            "decision_snapshot_status_counts": counts,
            "duration_seconds": time.monotonic() - started,
            "failure_accounting": accounting.to_dict(),
            "logical_decisions": len(decisions),
            "measured_mu": network.measured_mu,
            "oracle_synthetic_risk_labels": True,
            "policy": initialization["policy"],
            "protocol_version": PROTOCOL_VERSION,
            "run_id": platform.run_id,
            "semantic_metrics": semantic_metrics.to_dict(),
            "status": "success" if counts["succeeded"] == expected else "degraded",
            "strategy": strategy,
            "teacher_samples_created": 0,
            "threshold_shadow_mode": "counterfactual_non_dispatching",
        }
        decisions.sort(key=lambda row: str(row["decision_id"]))
        shadow_rows.sort(
            key=lambda row: (str(row["decision_id"]), int(row["root_post_id"]))
        )
        v1._write_jsonl(directory / "per_decision.jsonl", decisions)
        v1._write_jsonl(directory / "threshold_shadow.jsonl", shadow_rows)
        v1._write_json(directory / "runtime_metrics.json", runtime)
        v1._write_json(directory / "unit_summary.json", summary)
        return decisions, summary, runtime
    finally:
        platform.db.close()


def _implementation_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _complete_payload(
    directory: Path,
    *,
    unit_id: str,
    backend: BackendName,
    config_sha256: str,
    expected: int,
    decisions: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    files = (
        "experiment.db",
        "per_decision.jsonl",
        "threshold_shadow.jsonl",
        "runtime_metrics.json",
        "unit_summary.json",
        "response_cache_v2_1.sqlite3",
        "network.json",
        "exposure_policy.json",
    )
    with sqlite3.connect(directory / "experiment.db") as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        snapshots, succeeded = connection.execute(
            "SELECT COUNT(*), SUM(status = 'succeeded') "
            "FROM diffusionguard_decision_snapshot"
        ).fetchone()
    return {
        "actual_completed_decisions": int(summary["completed_decisions"]),
        "agent_role": "behavior_simulator",
        "backend": backend,
        "config_sha256": config_sha256,
        "database_integrity": str(integrity),
        "expected_logical_decisions": expected,
        "file_sha256": {name: v1._sha256(directory / name) for name in files},
        "implementation_sha256": _implementation_sha(),
        "protocol_version": PROTOCOL_VERSION,
        "snapshot_count": int(snapshots),
        "status": str(summary["status"]),
        "succeeded_snapshot_count": int(succeeded or 0),
        "unit_id": unit_id,
        "unique_decision_ids": len(
            {str(row["decision_id"]) for row in decisions}
        ),
    }


def _valid_complete(directory: Path, expected: int) -> bool:
    try:
        marker = v1._json(directory / "complete.json")
        if any(
            (
                marker.get("protocol_version") != PROTOCOL_VERSION,
                marker.get("agent_role") != "behavior_simulator",
                marker.get("expected_logical_decisions") != expected,
                marker.get("actual_completed_decisions") != expected,
                marker.get("snapshot_count") != expected,
                marker.get("succeeded_snapshot_count") != expected,
                marker.get("unique_decision_ids") != expected,
                marker.get("database_integrity") != "ok",
            )
        ):
            return False
        return all(
            (directory / name).is_file()
            and v1._sha256(directory / name) == digest
            for name, digest in marker["file_sha256"].items()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


async def run_joint_v2_1(
    config_path: Path,
    backend: BackendName,
    output: Path,
    *,
    semantic_micro_pilot: bool = False,
    llm_settings: LLMSettings | None = None,
    model_factory: Callable[
        [LLMSettings], ManagedModelBackend | None
    ] = create_llm_model,
) -> dict[str, Any]:
    config = _load_config(config_path)
    plan = build_plan(
        config_path,
        backend,
        output,
        semantic_micro_pilot=semantic_micro_pilot,
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "preregistered_config.json")
    config_sha = str(plan["config_sha256"])
    selections = _selection_map(
        config, semantic_micro_pilot=semantic_micro_pilot
    )
    networks = {
        spec.condition_id: spec
        for spec in (
            v1._generate_network(config, condition)
            for condition in config["network"]["conditions"]
        )
    }
    all_decisions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    started = time.monotonic()
    for condition in config["network"]["conditions"]:
        condition_id = str(condition["condition_id"])
        for strategy in v1.STRATEGIES:
            selected = (
                selections.get((condition_id, strategy), set())
                if selections is not None
                else None
            )
            expected = len(selected) if selected is not None else 10
            if expected == 0:
                continue
            relative = Path("units") / condition_id / strategy
            pending = output / ".pending" / f"{condition_id}--{strategy}"
            decisions, summary, runtime = await _run_unit(
                config=config,
                config_path=config_path,
                network=networks[condition_id],
                strategy=strategy,
                backend=backend,
                directory=pending,
                selected=selected,
                llm_settings=llm_settings,
                model_factory=model_factory,
            )
            directory = output / relative
            directory.parent.mkdir(parents=True, exist_ok=True)
            pending.replace(directory)
            marker = _complete_payload(
                directory,
                unit_id=f"{condition_id}--{strategy}",
                backend=backend,
                config_sha256=config_sha,
                expected=expected,
                decisions=decisions,
                summary=summary,
            )
            v1._write_json(directory / "complete.json", marker)
            all_decisions.extend(decisions)
            summaries.append(summary)
            runtimes.append(runtime)
            units.append(
                {
                    "cache_path": str(
                        relative / "response_cache_v2_1.sqlite3"
                    ),
                    "complete": _valid_complete(directory, expected),
                    "condition_id": condition_id,
                    "logical_decisions": expected,
                    "measured_mu": networks[condition_id].measured_mu,
                    "relative_directory": str(relative),
                    "run_id": summary["run_id"],
                    "strategy": strategy,
                    "unit_id": f"{condition_id}--{strategy}",
                }
            )
    all_decisions.sort(key=lambda row: str(row["decision_id"]))
    summaries.sort(key=lambda row: (str(row["condition_id"]), str(row["strategy"])))
    accounting = Counter()
    runtime_totals = Counter()
    for summary in summaries:
        accounting.update(summary["failure_accounting"])
    runtime_keys = (
        "cache_hits",
        "first_attempt_count",
        "first_attempt_success_count",
        "json_validate_failed_count",
        "physical_backend_attempts",
        "physical_remote_attempts",
        "structured_retry_attempt_count",
        "structured_retry_failure_count",
        "structured_retry_success_count",
        *SemanticAuditMetrics.__dataclass_fields__,
    )
    for runtime in runtimes:
        for key in runtime_keys:
            runtime_totals[key] += int(runtime.get(key, 0))
    reliability = {
        "backend": backend,
        **{key: int(accounting[key]) for key in FailureAccounting.__dataclass_fields__},
        **{
            key: int(runtime_totals[key])
            for key in SemanticAuditMetrics.__dataclass_fields__
        },
        "cache_hits": int(runtime_totals["cache_hits"]),
        "complete_units": sum(bool(unit["complete"]) for unit in units),
        "first_attempt_count": int(runtime_totals["first_attempt_count"]),
        "first_attempt_success_count": int(
            runtime_totals["first_attempt_success_count"]
        ),
        "first_attempt_success_rate": (
            runtime_totals["first_attempt_success_count"]
            / runtime_totals["first_attempt_count"]
            if runtime_totals["first_attempt_count"]
            else 0.0
        ),
        "logical_decision_count": len(all_decisions),
        "physical_backend_attempts": int(
            runtime_totals["physical_backend_attempts"]
        ),
        "physical_remote_attempts": int(
            runtime_totals["physical_remote_attempts"]
        ),
        "provider_json_validate_failed_attempt_count": int(
            runtime_totals["json_validate_failed_count"]
        ),
        "structured_retry_attempt_count": int(
            runtime_totals["structured_retry_attempt_count"]
        ),
        "structured_retry_failure_count": int(
            runtime_totals["structured_retry_failure_count"]
        ),
        "structured_retry_success_count": int(
            runtime_totals["structured_retry_success_count"]
        ),
    }
    behavior = v1._aggregate_behavior(summaries)
    behavior["agent_role"] = "behavior_simulator"
    behavior["protocol_version"] = PROTOCOL_VERSION
    paired = v1._paired_comparisons(behavior)
    manifest = {
        "agent_role": "behavior_simulator",
        "backend": backend,
        "config_sha256": config_sha,
        "experiment_id": config["experiment_id"],
        "implementation_sha256": _implementation_sha(),
        "logical_decisions": len(all_decisions),
        "maximum_physical_requests_total": int(
            plan["maximum_physical_requests_total"]
        ),
        "semantic_micro_pilot": semantic_micro_pilot,
        "oracle_synthetic_risk_labels": True,
        "protocol_version": PROTOCOL_VERSION,
        "remote_api_calls": int(reliability["physical_remote_attempts"]),
        "teacher_or_training_samples_created": 0,
        "units": units,
    }
    v1._write_json(output / "manifest.json", manifest)
    v1._write_jsonl(output / "per_decision.jsonl", all_decisions)
    v1._write_json(output / "reliability_summary.json", reliability)
    v1._write_json(output / "behavior_propagation_summary.json", behavior)
    v1._write_json(output / "paired_comparisons.json", paired)
    integrity = v1._audit_outputs(output, units, all_decisions)
    refresh_trace_count = 0
    impression_count = 0
    for unit in units:
        unit_directory = output / str(unit["relative_directory"])
        with sqlite3.connect(unit_directory / "experiment.db") as connection:
            refresh_trace_count += int(
                connection.execute(
                    "SELECT COUNT(*) FROM trace WHERE action = 'refresh'"
                ).fetchone()[0]
            )
            impression_count += int(
                connection.execute(
                    "SELECT COUNT(*) FROM diffusionguard_impression"
                ).fetchone()[0]
            )
    integrity.update(
        {
            "complete_unit_count": reliability["complete_units"],
            "cross_protocol_cache_pollution_detected": False,
            "cross_strategy_cache_pollution_detected": False,
            "protocol_cache_isolation_passed": len(
                {str(unit["cache_path"]) for unit in units}
            )
            == len(units),
            "root_duplicate_report_or_repost": v2._root_duplicates(
                all_decisions
            ),
            "semantic_inconsistent_success_count": sum(
                row.get("status") == "succeeded"
                and not row.get("semantic_consistent", False)
                for row in all_decisions
            ),
            "refresh_trace_count": refresh_trace_count,
            "impression_count": impression_count,
            "expected_refresh_trace_count": len(all_decisions),
            "semantic_correction_extra_refresh_count": max(
                0, refresh_trace_count - len(all_decisions)
            ),
            "teacher_or_training_samples_created": 0,
        }
    )
    integrity["one_refresh_per_logical_decision"] = (
        refresh_trace_count == len(all_decisions)
    )
    integrity["semantic_correction_added_impression_batch"] = not integrity[
        "one_refresh_per_logical_decision"
    ]
    integrity["root_action_deduplication_passed"] = not integrity[
        "root_duplicate_report_or_repost"
    ]
    integrity["semantic_dispatch_guard_passed"] = (
        integrity["semantic_inconsistent_success_count"] == 0
    )
    integrity["physical_request_limit_enforced"] = (
        int(reliability["physical_backend_attempts"])
        <= int(plan["maximum_physical_requests_total"])
    )
    integrity["expected_exposure_differences_observed"] = (
        v1._expected_exposure_differences(behavior)
        if not semantic_micro_pilot
        else True
    )
    v1._write_json(output / "integrity_audit.json", integrity)
    role_audits = v2._role_audits(all_decisions)
    v1._write_json(output / "role_quality_audits.json", role_audits)
    ready = all(
        (
            accounting["completed_decision_count"] == len(all_decisions),
            accounting["unrecovered_logical_error_count"] == 0,
            reliability["complete_units"] == len(units),
            integrity["trace_binding_rate"] == 1.0,
            integrity["feed_consistency_rate"] == 1.0,
            integrity["policy_impression_consistency_rate"] == 1.0,
            integrity["threshold_shadow_non_mutating"],
            integrity["root_action_deduplication_passed"],
            integrity["protocol_cache_isolation_passed"],
            integrity["semantic_dispatch_guard_passed"],
            integrity["one_refresh_per_logical_decision"],
            integrity["label_leakage_check_passed"],
            integrity["secret_scan_passed"],
            integrity["physical_request_limit_enforced"],
        )
    )
    status = (
        "success"
        if ready
        else ("degraded" if accounting["completed_decision_count"] else "failed")
    )
    summary = {
        "agent_role": "behavior_simulator",
        "backend": backend,
        "completed_decisions": int(accounting["completed_decision_count"]),
        "duration_seconds": time.monotonic() - started,
        "exit_code": v1.EXIT_CODES[status],
        "fake_validation_ready_for_semantic_micro_pilot": bool(
            ready and backend == "fake" and not semantic_micro_pilot
        ),
        "logical_decisions": len(all_decisions),
        "semantic_micro_pilot": semantic_micro_pilot,
        "oracle_synthetic_risk_labels": True,
        "physical_remote_attempts": int(
            reliability["physical_remote_attempts"]
        ),
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "teacher_or_training_samples_created": 0,
    }
    v1._write_json(output / "summary.json", summary)
    return {
        "behavior": behavior,
        "integrity": integrity,
        "manifest": manifest,
        "paired_comparisons": paired,
        "reliability": reliability,
        "role_audits": role_audits,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("fake", "groq"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--semantic-micro-pilot", action="store_true")
    parser.add_argument("--confirm-remote-run", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plan = build_plan(
            args.config,
            args.backend,
            args.output,
            semantic_micro_pilot=args.semantic_micro_pilot,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"joint v2.1 configuration failed: {exc}") from None
    if args.plan:
        print(v1._canonical(plan, pretty=True), end="")
        return
    if args.backend == "groq" and not args.semantic_micro_pilot:
        raise SystemExit("v2.1 groq backend requires --semantic-micro-pilot")
    if args.backend == "groq" and not args.confirm_remote_run:
        raise SystemExit(
            "backend=groq requires --confirm-remote-run before environment loading"
        )
    settings = None
    if args.backend == "groq":
        if not args.env_file.is_file():
            raise SystemExit("groq backend requires an explicit local env file")
        load_dotenv(args.env_file, override=False)
        settings = LLMSettings.from_env()
        if not settings.enabled or settings.provider != "groq":
            raise SystemExit("groq backend requires enabled provider=groq settings")
    try:
        result = asyncio.run(
            run_joint_v2_1(
                args.config,
                args.backend,
                args.output,
                semantic_micro_pilot=args.semantic_micro_pilot,
                llm_settings=settings,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(f"joint v2.1 experiment failed: {exc}") from None
    print(v1._canonical(result["summary"], pretty=True), end="")
    raise SystemExit(int(result["summary"]["exit_code"]))


__all__ = [
    "PROTOCOL_VERSION",
    "SEMANTIC_CORRECTION_PROMPT",
    "ChoiceActionMismatch",
    "ChoiceTargetMismatch",
    "FakeSemanticModel",
    "QuoteTextContractFailure",
    "SemanticActionMaskBuilder",
    "SemanticActionResponse",
    "SemanticAuditMetrics",
    "build_plan",
    "build_semantic_messages",
    "parse_semantic_response",
    "request_semantic_with_correction",
    "run_joint_v2_1",
    "semantic_response_format",
    "validate_semantic_binding",
]


if __name__ == "__main__":
    main()
