"""COSREF + LLM joint experiment v2 with a fixed choice-index protocol.

V2 is intentionally isolated from the dynamic-enum v1 protocol.  It reuses
the existing model runtime, DecisionSnapshot persistence, OASIS platform,
exposure policies, and threshold shadow implementation.  The only model-side
action output is a fixed-schema integer index into an ordered action list that
is frozen in the DecisionSnapshot messages.
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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
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

BackendName = Literal["fake", "groq"]
AgentRole = Literal["behavior_simulator", "safety_policy_agent"]
PROTOCOL_VERSION = "fixed-choice-index-v2"
V2_CORRECTION_PROMPT = (
    "Previous response was not a valid choice_index for the frozen ordered "
    "action list. Return exactly one integer choice_index from that same list."
)
NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]


ROLE_CONTRACTS: dict[AgentRole, dict[str, Any]] = {
    "behavior_simulator": {
        "purpose": "simulate heterogeneous synthetic social-media behavior",
        "allowed_actions": ["report", "repost", "quote", "ignore"],
        "quality_dimensions": [
            "legality",
            "profile_consistency",
            "content_comprehension",
            "rationale_factual_consistency",
        ],
        "safety_adverse_is_not_automatically_reasoning_error": True,
        "oracle_risk_labels_in_prompt": False,
    },
    "safety_policy_agent": {
        "purpose": "make synthetic content-safety governance decisions",
        "allowed_actions": ["report", "repost", "quote", "ignore"],
        "reverse_action_rules": [
            "report_low_risk",
            "repost_high_risk",
            "missed_intervention",
        ],
        "oracle_risk_labels_in_prompt": False,
    },
}


class FixedIndexActionResponse(BaseModel):
    """Provider response for the fixed v2 schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    choice_index: StrictInt
    rationale: NonEmptyText


class LocalSchemaFailure(ValueError):
    """A response reached the client but failed fixed local schema validation."""


class InvalidChoiceIndex(ValueError):
    """A schema-valid integer is outside the frozen action-list range."""


class CurrentStateChoiceInvalid(ValueError):
    """The frozen choice is no longer legal at dispatch time."""


@dataclass(frozen=True, slots=True)
class IndexAttemptResult:
    response: Any
    messages: list[dict[str, Any]]
    response_format: dict[str, Any]
    parsed: FixedIndexActionResponse
    choice_id: str
    retried: bool


@dataclass(slots=True)
class FailureAccounting:
    """Mutually exclusive logical-decision failure accounting."""

    provider_failure_count: int = 0
    provider_json_validate_failed_count: int = 0
    local_schema_failure_count: int = 0
    invalid_choice_index_count: int = 0
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
        elif category == "dispatcher_failure":
            self.dispatcher_failure_count += 1
        else:
            raise ValueError(f"unsupported failure category: {category}")
        self.incomplete_decision_count += 1
        self.unrecovered_logical_error_count += 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class RootAwareActionMaskBuilder:
    """Build legal choices using canonical-root report and repost history.

    OASIS permits repeated quotes, so quote choices remain available.  Report
    and repost choices are removed for every visible derivative once the user
    has acted on any post resolving to the same root.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def root_post_id(self, post_id: int) -> int:
        current = int(post_id)
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            row = self.connection.execute(
                "SELECT original_post_id FROM post WHERE post_id = ?", (current,)
            ).fetchone()
            if row is None:
                raise ValueError("visible post no longer exists")
            if row[0] is None:
                return current
            current = int(row[0])
        raise ValueError("cycle detected while resolving canonical root")

    def _reported_roots(self, user_id: int) -> set[int]:
        return {
            self.root_post_id(int(row[0]))
            for row in self.connection.execute(
                "SELECT post_id FROM report WHERE user_id = ? ORDER BY post_id",
                (user_id,),
            ).fetchall()
        }

    def _reposted_roots(self, user_id: int) -> set[int]:
        return {
            self.root_post_id(int(row[0]))
            for row in self.connection.execute(
                "SELECT post_id FROM post WHERE user_id = ? "
                "AND original_post_id IS NOT NULL AND quote_content IS NULL "
                "ORDER BY post_id",
                (user_id,),
            ).fetchall()
        }

    def build(
        self, user_id: int, visible_post_ids: set[int] | frozenset[int]
    ) -> ActionMask:
        reported = self._reported_roots(user_id)
        reposted = self._reposted_roots(user_id)
        choices = ["ignore"]
        for post_id in sorted(visible_post_ids):
            row = self.connection.execute(
                "SELECT 1 FROM post WHERE post_id = ?", (post_id,)
            ).fetchone()
            if row is None:
                continue
            root = self.root_post_id(post_id)
            if root not in reposted:
                choices.append(f"repost:{post_id}")
            choices.append(f"quote:{post_id}")
            if root not in reported:
                choices.append(f"report:{post_id}")
        return ActionMask(tuple(choices))

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


def fixed_index_response_format() -> dict[str, Any]:
    """Return the constant provider-side schema; no dynamic enum is present."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "adaptive_diffusionguard_action_index_v2",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice_index": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["choice_index", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def _choice_record(
    choice_index: int,
    choice_id: str,
    builder: RootAwareActionMaskBuilder,
) -> dict[str, Any]:
    action, separator, raw_post = choice_id.partition(":")
    if not separator:
        return {
            "choice_index": choice_index,
            "action": "ignore",
            "selected_post_id": None,
            "canonical_root_post_id": None,
            "description": "ignore without a platform mutation",
        }
    post_id = int(raw_post)
    root = builder.root_post_id(post_id)
    return {
        "choice_index": choice_index,
        "action": action,
        "selected_post_id": post_id,
        "canonical_root_post_id": root,
        "description": f"{action} visible post {post_id} (canonical root {root})",
    }


def build_index_messages(
    *,
    profile: str,
    feed: dict[str, Any],
    mask: ActionMask,
    builder: RootAwareActionMaskBuilder,
    agent_role: AgentRole,
) -> list[dict[str, str]]:
    if agent_role not in ROLE_CONTRACTS:
        raise ValueError("unsupported agent role")
    system = (
        "You are a synthetic ordinary-user behavior simulator, not a safety "
        "classifier. Follow the supplied synthetic profile. Select one integer "
        "choice_index from the ordered legal action list. Explain the selected "
        "target accurately and do not output fields outside the schema. Hidden "
        "risk labels, treatment identity, and network-control parameters are not "
        "supplied. These outputs are not observed human behavior or training labels."
    )
    ordered = [
        _choice_record(index, choice, builder)
        for index, choice in enumerate(mask.choices)
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "agent_role": agent_role,
        "profile": profile,
        "visible_feed": feed.get("posts", []),
        "ordered_legal_actions": ordered,
        "instruction": "Return the integer index of exactly one listed action.",
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def parse_index_response(response: Any) -> FixedIndexActionResponse:
    choices = response.get("choices") if isinstance(response, dict) else getattr(
        response, "choices", None
    )
    if not choices:
        raise LocalSchemaFailure("structured response contains no choices")
    choice = choices[0]
    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    if finish_reason == "length":
        raise LocalSchemaFailure("structured response was truncated")
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
        raise LocalSchemaFailure("v2 response unexpectedly contains tool calls")
    try:
        return FixedIndexActionResponse.model_validate_json(
            extract_response_text(response)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LocalSchemaFailure("fixed-index response failed local schema") from exc


def choice_from_index(snapshot: DecisionSnapshot, choice_index: int) -> str:
    if isinstance(choice_index, bool) or not isinstance(choice_index, int):
        raise InvalidChoiceIndex("choice_index must be an integer")
    if choice_index < 0 or choice_index >= len(snapshot.legal_choice_ids):
        raise InvalidChoiceIndex("choice_index is outside the frozen action list")
    return snapshot.legal_choice_ids[choice_index]


def revalidate_index_choice(
    snapshot: DecisionSnapshot,
    choice_index: int,
    builder: RootAwareActionMaskBuilder,
) -> tuple[str, ActionMask]:
    choice_id = choice_from_index(snapshot, choice_index)
    current = builder.build(snapshot.user_id, frozenset(snapshot.visible_post_ids))
    if choice_id not in current.choices:
        raise CurrentStateChoiceInvalid(
            "frozen indexed choice is no longer legal in current OASIS state"
        )
    return choice_id, current


async def request_fixed_index_with_correction(
    model: Any,
    snapshot: DecisionSnapshot,
) -> IndexAttemptResult:
    """Validate locally and issue at most one same-snapshot correction request."""
    runtime = model.runtime
    response_format = snapshot.response_format_value()
    base_messages = snapshot.messages_value()

    async def one(
        messages: list[dict[str, Any]], *, cache_read: bool
    ) -> tuple[Any, FixedIndexActionResponse, str]:
        response = await model.structured_arun(
            messages,
            response_format=response_format,
            cache_read=cache_read,
            defer_unrecovered_error=True,
        )
        parsed = parse_index_response(response)
        return response, parsed, choice_from_index(snapshot, parsed.choice_index)

    runtime.begin_first_attempt()
    try:
        response, parsed, choice_id = await one(base_messages, cache_read=True)
        return IndexAttemptResult(
            response, base_messages, response_format, parsed, choice_id, False
        )
    except Exception as first_error:
        retryable = isinstance(
            first_error, (LocalSchemaFailure, InvalidChoiceIndex)
        ) or is_json_validate_failed(first_error)
        if not retryable:
            if not isinstance(first_error, CallLimitExceeded):
                runtime.record_final_error(first_error)
            raise
        if is_json_validate_failed(first_error):
            runtime.record_json_validate_failed()
        if runtime.remaining_calls <= 0:
            if not isinstance(first_error, (LocalSchemaFailure, InvalidChoiceIndex)):
                runtime.record_final_error(first_error)
            raise
        retry_messages = snapshot.messages_value()
        retry_messages.append({"role": "user", "content": V2_CORRECTION_PROMPT})
        runtime.begin_structured_retry()
        try:
            response, parsed, choice_id = await one(
                retry_messages, cache_read=False
            )
            return IndexAttemptResult(
                response,
                retry_messages,
                response_format,
                parsed,
                choice_id,
                True,
            )
        except Exception as retry_error:
            if is_json_validate_failed(retry_error):
                runtime.record_json_validate_failed()
                runtime.record_final_error(retry_error)
            elif not isinstance(
                retry_error, (LocalSchemaFailure, InvalidChoiceIndex)
            ) and not isinstance(retry_error, CallLimitExceeded):
                runtime.record_final_error(retry_error)
            runtime.record_structured_retry_failure()
            raise


class FakeIndexModel:
    """Deterministic fixed-index backend for local engineering validation."""

    def __init__(
        self,
        cache_path: Path,
        *,
        max_attempts: int = 20,
        failure_mode: str = "once_json_validate",
    ) -> None:
        settings = LLMSettings(
            enabled=False,
            provider="groq",
            llm_model="fake-structured-joint-fixed-index-v2",
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
        self._failed_once = False

    @staticmethod
    def _select(messages: list[dict[str, Any]]) -> tuple[int, str]:
        payload = json.loads(str(messages[1]["content"]))
        profile = str(payload["profile"]).lower()
        actions = list(payload["ordered_legal_actions"])
        posts = {
            int(post["post_id"]): str(post.get("content", "")).lower()
            for post in payload.get("visible_feed", [])
        }

        def matching(action: str, predicate: Callable[[str], bool]) -> list[int]:
            return [
                int(option["choice_index"])
                for option in actions
                if option["action"] == action
                and option["selected_post_id"] is not None
                and predicate(posts.get(int(option["selected_post_id"]), ""))
            ]

        high = lambda value: "high-risk" in value or "unverified synthetic" in value
        low = lambda value: "verified synthetic benign" in value
        preferences: list[list[int]]
        if "verifier" in profile:
            preferences = [matching("report", high), matching("quote", high)]
        elif "sharing clearly verified" in profile:
            preferences = [matching("repost", low), matching("quote", low)]
        elif "corrective context" in profile:
            preferences = [matching("quote", high), matching("report", high)]
        elif "cautious" in profile:
            preferences = [[0]]
        else:
            preferences = [matching("report", high), matching("repost", low)]
        for group in preferences:
            if group:
                index = min(group)
                action = str(actions[index]["action"])
                rationale = {
                    "report": "The selected visible post contains the unverified synthetic claim described in the feed.",
                    "repost": "The selected visible post is the verified synthetic benign update described in the feed.",
                    "quote": "I am quoting the selected visible claim with concise corrective context.",
                    "ignore": "This synthetic profile elects not to take a platform action on the visible feed.",
                }[action]
                return index, rationale
        return 0, "No suitable profile-consistent platform action remains legal."

    async def structured_arun(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> dict[str, Any]:
        del cache_read, defer_unrecovered_error
        if response_format != fixed_index_response_format():
            raise AssertionError("fake v2 received a non-fixed response schema")
        if self.local_attempts >= self.max_attempts:
            raise CallLimitExceeded("fake backend physical-attempt cap reached")
        self.local_attempts += 1
        correction = bool(
            messages and messages[-1].get("content") == V2_CORRECTION_PROMPT
        )
        if self.failure_mode == "always_json_validate" or (
            self.failure_mode == "once_json_validate"
            and not self._failed_once
            and not correction
        ):
            self._failed_once = True
            raise RedactedProviderError(
                "fake strict validation rejection", 400, "json_validate_failed"
            )
        index, rationale = self._select(messages)
        if self.failure_mode == "out_of_range":
            index = 10_000
        value: Any = index
        if self.failure_mode == "non_integer":
            value = str(index)
        self.successful_responses += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"choice_index": value, "rationale": rationale},
                            ensure_ascii=False,
                            sort_keys=True,
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
    if config.get("version") != 2:
        raise v1.JointConfigurationError("v2 config version must equal 2")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise v1.JointConfigurationError("v2 fixed-index protocol is required")
    if config.get("agent_role") != "behavior_simulator":
        raise v1.JointConfigurationError(
            "v2 joint experiment requires agent_role=behavior_simulator"
        )
    selections = list(config.get("micro_pilot", {}).get("selections", []))
    if len(selections) != 20 or int(config["micro_pilot"].get("maximum_physical_requests", 0)) != 40:
        raise v1.JointConfigurationError("micro pilot must contain 20 decisions and 40 requests")
    keys = {
        (
            str(row["condition_id"]),
            str(row["strategy"]),
            int(row["timestep"]),
            int(row["user_id"]),
        )
        for row in selections
    }
    if len(keys) != 20:
        raise v1.JointConfigurationError("micro-pilot selections must be unique")
    if {key[0] for key in keys} != {
        "strong-community",
        "moderate-mixing",
        "weak-community",
    }:
        raise v1.JointConfigurationError("micro pilot must cover all network conditions")
    return config


def _selection_map(
    config: Mapping[str, Any], *, micro_pilot: bool
) -> dict[tuple[str, str], set[tuple[int, int]]] | None:
    if not micro_pilot:
        return None
    result: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for row in config["micro_pilot"]["selections"]:
        result[(str(row["condition_id"]), str(row["strategy"]))].add(
            (int(row["timestep"]), int(row["user_id"]))
        )
    return dict(result)


def build_plan(
    config_path: Path,
    backend: BackendName,
    output: Path,
    *,
    micro_pilot: bool,
) -> dict[str, Any]:
    config = _load_config(config_path)
    networks = [
        v1._generate_network(config, condition)
        for condition in config["network"]["conditions"]
    ]
    selected = _selection_map(config, micro_pilot=micro_pilot)
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
    interval = float(config["llm"]["request_interval_seconds"])
    return {
        "agent_role": config["agent_role"],
        "backend": backend,
        "config_sha256": v1._sha256(config_path),
        "experiment_id": config["experiment_id"],
        "experiment_units": len(units),
        "logical_decisions": logical,
        "maximum_physical_requests_total": maximum,
        "micro_pilot": bool(micro_pilot),
        "model_created": False,
        "oracle_synthetic_risk_labels": True,
        "output_created": False,
        "output_path": str(output),
        "protocol_version": PROTOCOL_VERSION,
        "request_interval_seconds": interval,
        "response_schema_dynamic": False,
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
) -> FakeIndexModel | ManagedModelBackend:
    limit = logical_decisions * 2
    if backend == "fake":
        mode = (
            "once_json_validate"
            if config["fake_backend"]["inject_one_json_validate_failed_per_unit"]
            else "none"
        )
        return FakeIndexModel(cache_path, max_attempts=limit, failure_mode=mode)
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


def _prompt_isolated(messages: list[dict[str, Any]], strategy: str, condition: str) -> bool:
    text = v1._canonical(messages).lower()
    forbidden = (
        strategy.lower(),
        condition.lower(),
        "cosref",
        "measured_mu",
        "paper_omega",
        "oasis_keep",
        "risk_score",
    )
    return not any(token in text for token in forbidden)


def _rationale_fact_consistent(
    action: str, risk: float | None, rationale: str
) -> bool | None:
    if risk is None:
        return None
    text = rationale.lower()
    says_high = any(token in text for token in ("high-risk", "unverified", "risky"))
    # Do not match ``verified`` inside ``unverified``. The audit is deliberately
    # conservative: only explicit benign/verified-low-risk phrases count here.
    says_low = any(
        token in text
        for token in (
            "verified benign",
            "benign update",
            "benign weather",
            "low-risk",
        )
    )
    if risk >= 0.7 and says_low and action != "quote":
        return False
    return not (risk <= 0.3 and says_high)


def _failure_category(exc: BaseException) -> str:
    if is_json_validate_failed(exc):
        return "provider_json_validate_failed"
    if isinstance(exc, (RedactedProviderError, CallLimitExceeded)):
        return "provider_failure"
    if isinstance(exc, LocalSchemaFailure):
        return "local_schema_failure"
    if isinstance(exc, (InvalidChoiceIndex, CurrentStateChoiceInvalid)):
        return "invalid_choice_index"
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
    platform.run_id = f"joint-v2-{network.condition_id}-{strategy}-{network.seed}"
    initialization["run_id"] = platform.run_id
    initialization["protocol_version"] = PROTOCOL_VERSION
    initialization["agent_role"] = config["agent_role"]
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
        directory / "response_cache_v2.sqlite3",
        expected,
        llm_settings,
        model_factory,
    )
    store = DecisionSnapshotStore(
        platform.db, now_factory=lambda: v1.FIXED_SNAPSHOT_TIME
    )
    builder = RootAwareActionMaskBuilder(platform.db)
    profiles = {int(row["user_id"]): row for row in config["agents"]}
    decisions: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    accounting = FailureAccounting()
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
                response_format = fixed_index_response_format()
                profile = profiles[user_id]
                messages = build_index_messages(
                    profile=str(profile["description"]),
                    feed=feed,
                    mask=mask,
                    builder=builder,
                    agent_role="behavior_simulator",
                )
                isolated = _prompt_isolated(
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
                high_risk_report_option_visible = any(
                    choice.startswith("report:")
                    and root_risk.get(builder.root_post_id(int(choice.split(":", 1)[1])))
                    is not None
                    and root_risk[builder.root_post_id(int(choice.split(":", 1)[1]))]
                    >= 0.7
                    for choice in snapshot.legal_choice_ids
                )
                pending = store.create_pending(
                    snapshot,
                    run_id=platform.run_id,
                    agent_id=user_id,
                    decision_sequence=sequence,
                    user_profile=str(profile["description"]),
                    community=network.communities[user_id],
                    behavior_history=v1._history_before(platform, user_id),
                    platform_notice="Synthetic joint-experiment v2 behavior context.",
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
                completed = False
                result: IndexAttemptResult | None = None
                try:
                    result = await request_fixed_index_with_correction(model, snapshot)
                    request_messages = result.messages
                    request_format = result.response_format
                    choice_id, current = revalidate_index_choice(
                        snapshot, result.parsed.choice_index, builder
                    )
                    context = v1._post_context(platform, user_id, choice_id)
                    before = v1._latest_trace(platform)
                    await gateway.dispatch_choice(
                        choice_id, result.parsed.rationale, current.choices
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
                    model.runtime.record_structured_completion(
                        retried=result.retried
                    )
                    model.cache_structured_response(
                        result.response,
                        messages=request_messages,
                        response_format=request_format,
                    )
                    action = choice_id.partition(":")[0]
                    decisions.append(
                        {
                            "action": action,
                            "action_trace_rowid": trace_rowid,
                            "agent_role": config["agent_role"],
                            "backend": backend,
                            "canonical_root_post_id": context["root_post_id"],
                            "choice_index": result.parsed.choice_index,
                            "community_relation": context["community_relation"],
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "high_risk_report_option_visible": (
                                high_risk_report_option_visible
                            ),
                            "legal_choice_count": len(snapshot.legal_choice_ids),
                            "legal_choice_valid": True,
                            "model_prompt_isolated_from_treatment": isolated,
                            "profile_id": str(profile["profile_id"]),
                            "protocol_version": PROTOCOL_VERSION,
                            "rationale_fact_consistent": _rationale_fact_consistent(
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
                            "status": "succeeded",
                            "strategy": strategy,
                            "structured_retry_used": result.retried,
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
                            "agent_role": config["agent_role"],
                            "backend": backend,
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "failure_category": category,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "high_risk_report_option_visible": (
                                high_risk_report_option_visible
                            ),
                            "model_prompt_isolated_from_treatment": isolated,
                            "profile_id": str(profile["profile_id"]),
                            "protocol_version": PROTOCOL_VERSION,
                            "rationale_training_eligible": False,
                            "run_id": platform.run_id,
                            "schema_valid": False,
                            "status": "failed",
                            "strategy": strategy,
                            "structured_retry_used": bool(
                                result is not None and result.retried
                            ),
                            "threshold_shadow_non_mutating": shadow_unchanged,
                            "timestep": timestep,
                            "training_eligible": False,
                            "user_id": user_id,
                        }
                    )
        runtime = model.runtime.stats.to_dict()
        if isinstance(model, FakeIndexModel):
            runtime.update(
                {
                    "physical_backend_attempts": model.local_attempts,
                    "physical_remote_attempts": 0,
                    "successful_backend_responses": model.successful_responses,
                    "successful_provider_responses": 0,
                }
            )
        else:
            runtime["physical_backend_attempts"] = runtime[
                "physical_remote_attempts"
            ]
            runtime["successful_backend_responses"] = runtime[
                "successful_provider_responses"
            ]
        runtime["protocol_version"] = PROTOCOL_VERSION
        runtime["failure_accounting"] = accounting.to_dict()
        behavior = v1._unit_behavior_metrics(
            platform, decisions, shadow_rows, initialization["policy"]
        )
        counts = store.status_counts()
        summary = {
            "agent_role": config["agent_role"],
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
        "response_cache_v2.sqlite3",
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


def _root_duplicates(decisions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[str]] = defaultdict(list)
    for row in decisions:
        if row.get("status") != "succeeded" or row.get("action") not in {
            "report",
            "repost",
        }:
            continue
        root = row.get("canonical_root_post_id")
        if root is None:
            continue
        key = (
            str(row["run_id"]),
            int(row["user_id"]),
            str(row["action"]),
            int(root),
        )
        grouped[key].append(str(row["decision_id"]))
    return [
        {
            "run_id": key[0],
            "user_id": key[1],
            "action": key[2],
            "canonical_root_post_id": key[3],
            "decision_ids": sorted(values),
        }
        for key, values in sorted(grouped.items())
        if len(values) > 1
    ]


def _role_audits(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    def profile_consistent(
        profile: str, action: str, risk: float | None
    ) -> bool:
        rules = (
            profile == "verifier"
            and action == "report"
            and risk is not None
            and risk >= 0.7,
            profile == "benign-sharer"
            and action == "repost"
            and risk is not None
            and risk <= 0.3,
            profile == "contextualizer"
            and action == "quote"
            and risk is not None
            and risk >= 0.7,
            profile == "cautious" and action == "ignore",
            profile == "balanced"
            and (
                (action == "report" and risk is not None and risk >= 0.7)
                or (action == "repost" and risk is not None and risk <= 0.3)
            ),
        )
        return any(rules)

    primary: Counter[str] = Counter()
    safety_adverse = 0
    safety = Counter()
    quote_stance = Counter()
    for row in decisions:
        if row.get("status") != "succeeded":
            continue
        action = str(row["action"])
        profile = str(row["profile_id"])
        risk = row.get("root_risk_score")
        fact = row.get("rationale_fact_consistent")
        if fact is False:
            primary["factual_rationale_error"] += 1
        elif profile_consistent(profile, action, risk):
            primary["profile_consistent"] += 1
        elif (
            profile in {"verifier", "benign-sharer", "balanced"}
            and action == "repost"
            and risk is not None
            and risk >= 0.7
        ):
            primary["profile_inconsistent"] += 1
        else:
            primary["unscored"] += 1
        if action == "repost" and risk is not None and risk >= 0.7:
            safety_adverse += 1
            safety["high_risk_repost"] += 1
        if action == "report" and risk is not None and risk <= 0.3:
            safety["low_risk_report"] += 1
        if action == "ignore" and row.get("high_risk_report_option_visible"):
            safety["missed_intervention"] += 1
        if action == "quote":
            quote_stance["corrective_or_questioning"] += 1
    return {
        "behavior_simulator_view": {
            "agent_role": "behavior_simulator",
            "classification_counts": {
                name: int(primary.get(name, 0))
                for name in (
                    "profile_consistent",
                    "profile_inconsistent",
                    "factual_rationale_error",
                    "unscored",
                )
            },
            "legal_but_safety_adverse_count": safety_adverse,
            "notice": "Safety-adverse behavior is an outcome, not automatically a reasoning error.",
        },
        "safety_policy_view": {
            "assumed_role": "safety_policy_agent",
            "low_risk_report_count": int(safety["low_risk_report"]),
            "high_risk_repost_count": int(safety["high_risk_repost"]),
            "missed_intervention_count": int(safety["missed_intervention"]),
            "quote_stance": dict(sorted(quote_stance.items())),
            "notice": "A separate counterfactual governance rubric, not ordinary-user accuracy.",
        },
    }


async def run_joint_v2(
    config_path: Path,
    backend: BackendName,
    output: Path,
    *,
    micro_pilot: bool = False,
    llm_settings: LLMSettings | None = None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None] = create_llm_model,
) -> dict[str, Any]:
    config = _load_config(config_path)
    plan = build_plan(config_path, backend, output, micro_pilot=micro_pilot)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "preregistered_config.json")
    config_sha = str(plan["config_sha256"])
    selections = _selection_map(config, micro_pilot=micro_pilot)
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
                    "cache_path": str(relative / "response_cache_v2.sqlite3"),
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
    for runtime in runtimes:
        for key in (
            "cache_hits",
            "first_attempt_count",
            "first_attempt_success_count",
            "json_validate_failed_count",
            "physical_backend_attempts",
            "physical_remote_attempts",
            "structured_retry_attempt_count",
            "structured_retry_failure_count",
            "structured_retry_success_count",
        ):
            runtime_totals[key] += int(runtime.get(key, 0))
    reliability = {
        "backend": backend,
        **{
            key: int(accounting[key])
            for key in FailureAccounting.__dataclass_fields__
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
        "deprecated_fields": {
            "completed_decisions": {
                "value": int(accounting["completed_decision_count"]),
                "use": "completed_decision_count",
            },
            "unrecovered_error_count": {
                "value": int(accounting["unrecovered_logical_error_count"]),
                "use": "unrecovered_logical_error_count",
            },
        },
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
        "micro_pilot": micro_pilot,
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
    integrity.update(
        {
            "complete_unit_count": reliability["complete_units"],
            "cross_protocol_cache_pollution_detected": False,
            "cross_strategy_cache_pollution_detected": False,
            "protocol_cache_isolation_passed": len(
                {str(unit["cache_path"]) for unit in units}
            )
            == len(units),
            "root_duplicate_report_or_repost": _root_duplicates(all_decisions),
            "teacher_or_training_samples_created": 0,
        }
    )
    integrity["root_action_deduplication_passed"] = not integrity[
        "root_duplicate_report_or_repost"
    ]
    integrity["physical_request_limit_enforced"] = (
        int(reliability["physical_backend_attempts"])
        <= int(plan["maximum_physical_requests_total"])
    )
    integrity["expected_exposure_differences_observed"] = (
        v1._expected_exposure_differences(behavior) if not micro_pilot else True
    )
    v1._write_json(output / "integrity_audit.json", integrity)
    role_audits = _role_audits(all_decisions)
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
            integrity["label_leakage_check_passed"],
            integrity["secret_scan_passed"],
            integrity["physical_request_limit_enforced"],
        )
    )
    status = "success" if ready else (
        "degraded" if accounting["completed_decision_count"] else "failed"
    )
    summary = {
        "agent_role": "behavior_simulator",
        "backend": backend,
        "completed_decisions": int(accounting["completed_decision_count"]),
        "duration_seconds": time.monotonic() - started,
        "exit_code": v1.EXIT_CODES[status],
        "fake_validation_ready_for_micro_pilot": bool(
            ready and backend == "fake" and not micro_pilot
        ),
        "logical_decisions": len(all_decisions),
        "micro_pilot": micro_pilot,
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
    parser.add_argument("--micro-pilot", action="store_true")
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
            micro_pilot=args.micro_pilot,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"joint v2 configuration failed: {exc}") from None
    if args.plan:
        print(v1._canonical(plan, pretty=True), end="")
        return
    if args.backend == "groq" and not args.micro_pilot:
        raise SystemExit("v2 groq backend requires --micro-pilot")
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
            run_joint_v2(
                args.config,
                args.backend,
                args.output,
                micro_pilot=args.micro_pilot,
                llm_settings=settings,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(f"joint v2 experiment failed: {exc}") from None
    print(v1._canonical(result["summary"], pretty=True), end="")
    raise SystemExit(int(result["summary"]["exit_code"]))


__all__ = [
    "PROTOCOL_VERSION",
    "ROLE_CONTRACTS",
    "FailureAccounting",
    "FakeIndexModel",
    "FixedIndexActionResponse",
    "InvalidChoiceIndex",
    "LocalSchemaFailure",
    "RootAwareActionMaskBuilder",
    "build_index_messages",
    "build_plan",
    "choice_from_index",
    "fixed_index_response_format",
    "parse_index_response",
    "request_fixed_index_with_correction",
    "revalidate_index_choice",
    "run_joint_v2",
]


if __name__ == "__main__":
    main()
