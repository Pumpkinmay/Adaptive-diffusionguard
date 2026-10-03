"""Constrained COSREF + structured-action joint experiment v1.

The exposure policy is the only treatment.  The threshold process is a
read-only counterfactual shadow diagnostic and never dispatches an action.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dotenv import load_dotenv

from adaptive_diffusionguard.governance.cosref import (
    NoInterventionPolicy,
    StaticCOSREFPolicy,
)
from adaptive_diffusionguard.llm.action_gateway import ActionDecisionGateway
from adaptive_diffusionguard.llm.model_factory import LLMSettings, create_llm_model
from adaptive_diffusionguard.llm.runtime import (
    LLMRuntime,
    ManagedModelBackend,
    RedactedProviderError,
    ResponseCache,
)
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMaskBuilder,
    DecisionSnapshot,
    build_decision_messages,
    build_response_format,
    parse_structured_response,
    revalidate_snapshot_choice,
)
from adaptive_diffusionguard.llm.structured_retry import (
    CORRECTION_PROMPT,
    StructuredAttemptResult,
    request_with_structured_correction,
)
from adaptive_diffusionguard.recommendation.base import Candidate, CandidateGenerator
from adaptive_diffusionguard.storage.decision_snapshots import DecisionSnapshotStore
from adaptive_diffusionguard.theory.allocation import project_l1_cost
from adaptive_diffusionguard.theory.mixing import compute_mixing_statistics
from adaptive_diffusionguard.theory.threshold_response import (
    ThresholdResponseEngine,
    allocate_strict_oasis_keep,
)

if TYPE_CHECKING:
    from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

BackendName = Literal["fake", "groq"]
StrategyName = Literal[
    "no_intervention", "static_cosref", "theory_informed_cosref"
]
STRATEGIES: tuple[StrategyName, ...] = (
    "no_intervention",
    "static_cosref",
    "theory_informed_cosref",
)
ACTION_TRACE_NAMES = {
    "repost": "repost",
    "quote": "quote_post",
    "report": "report_post",
    "ignore": "do_nothing",
}
SECRET_PATTERN = re.compile(rb"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")
FIXED_SNAPSHOT_TIME = "2026-01-01T00:00:00.000000+00:00"
EXIT_CODES = {"success": 0, "degraded": 2, "failed": 1}


class JointConfigurationError(ValueError):
    """Configuration failure raised before environment or model access."""


class ResumeIntegrityError(RuntimeError):
    """A completed unit or its frozen configuration cannot be trusted."""


def _canonical(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options) + "\n"


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, _canonical(value, pretty=True))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(_canonical(dict(row)) for row in rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSONL object: {path}")
            rows.append(value)
    return rows


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    condition_id: str
    seed: int
    communities: dict[int, str]
    contacts: tuple[tuple[int, int], ...]
    directed_arcs: tuple[tuple[int, int], ...]
    measured_mu: float
    mixing: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "network_seed": self.seed,
            "communities": {str(k): v for k, v in sorted(self.communities.items())},
            "contacts": [list(edge) for edge in self.contacts],
            "directed_arcs": [list(edge) for edge in self.directed_arcs],
            "measured_mu": self.measured_mu,
            "mixing": self.mixing,
        }


def _condition_accepts_mu(condition: Mapping[str, Any], mu: float) -> bool:
    checks = (
        ("mu_min_inclusive", lambda value: mu >= value),
        ("mu_min_exclusive", lambda value: mu > value),
        ("mu_max_inclusive", lambda value: mu <= value),
        ("mu_max_exclusive", lambda value: mu < value),
    )
    return all(
        predicate(float(condition[key])) for key, predicate in checks if key in condition
    )


def _generate_network(config: Mapping[str, Any], condition: Mapping[str, Any]) -> NetworkSpec:
    node_count = int(config["network"]["nodes"])
    community_count = int(config["network"]["communities"])
    if node_count < 2 or community_count != 2:
        raise JointConfigurationError("v1 requires at least two nodes and two communities")
    block = node_count // community_count
    communities = {
        user_id: "community-a" if user_id < block else "community-b"
        for user_id in range(node_count)
    }
    seed = int(condition["network_seed"])
    rng = random.Random(seed)
    contacts: list[tuple[int, int]] = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            probability = (
                float(condition["p_intra"])
                if communities[left] == communities[right]
                else float(condition["p_inter"])
            )
            if rng.random() < probability:
                contacts.append((left, right))
    mixing = compute_mixing_statistics(
        contacts, communities, directed_input=False, directed_adaptation="symmetrize"
    )
    if mixing.mu is None or not _condition_accepts_mu(condition, mixing.mu):
        raise JointConfigurationError(
            f"network {condition['condition_id']} measured mu={mixing.mu} "
            "outside its preregistered interval"
        )
    directed = tuple(
        arc for left, right in contacts for arc in ((left, right), (right, left))
    )
    return NetworkSpec(
        condition_id=str(condition["condition_id"]),
        seed=seed,
        communities=communities,
        contacts=tuple(contacts),
        directed_arcs=directed,
        measured_mu=float(mixing.mu),
        mixing=mixing.as_dict(),
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = _json(path)
    if config.get("oracle_synthetic_risk_labels") is not True:
        raise JointConfigurationError("oracle_synthetic_risk_labels must be true")
    conditions = list(config.get("network", {}).get("conditions", []))
    strategies = list(config.get("exposure", {}).get("strategies", []))
    agents = list(config.get("agents", []))
    simulation = config.get("simulation", {})
    llm = config.get("llm", {})
    if len(conditions) != 3 or strategies != list(STRATEGIES) or len(agents) != 5:
        raise JointConfigurationError("v1 requires 3 conditions, 3 strategies, 5 agents")
    if int(simulation.get("timesteps", 0)) != 2:
        raise JointConfigurationError("v1 requires exactly two timesteps")
    orders = simulation.get("active_user_order", [])
    agent_ids = sorted(int(agent["user_id"]) for agent in agents)
    if len(orders) != 2 or any(sorted(map(int, order)) != agent_ids for order in orders):
        raise JointConfigurationError("each timestep must schedule all five agents once")
    expected = {
        "temperature": 0.0,
        "max_tokens": 512,
        "max_concurrency": 1,
        "max_logical_decisions_per_unit": 10,
        "max_physical_requests_per_unit": 20,
        "max_physical_requests_total": 180,
        "structured_correction_retries_per_decision": 1,
    }
    for key, value in expected.items():
        if llm.get(key) != value:
            raise JointConfigurationError(f"llm.{key} must equal {value!r}")
    if float(llm.get("request_interval_seconds", -1)) != 12.0:
        raise JointConfigurationError("request interval must remain 12 seconds")
    calibration = set()
    for condition in conditions:
        spec = _generate_network(config, condition)
        calibration.add(spec.condition_id)
        allocation = allocate_strict_oasis_keep(
            mu=spec.measured_mu,
            project_budget=float(config["exposure"]["project_l1_budget"]),
            keep_grid=[tuple(map(float, pair)) for pair in config["exposure"]["theory_candidates"]],
            tolerance=float(config["exposure"]["direction_tolerance"]),
            minimum_strict_gap=float(config["exposure"]["minimum_strict_gap"]),
        )
        chosen = tuple(
            map(
                float,
                config["exposure"]["theory_selected_by_direction"][allocation.direction],
            )
        )
        if chosen not in allocation.eligible_oasis_keep_pairs:
            raise JointConfigurationError(
                f"theory keep pair is invalid for actual mu={spec.measured_mu}"
            )
    if len(calibration) != 3:
        raise JointConfigurationError("condition IDs must be unique")
    return config


def build_plan(config_path: Path, backend: BackendName, output: Path, *, resume: bool) -> dict[str, Any]:
    """Read only configuration and derive a non-sensitive execution plan."""
    config = _load_config(config_path)
    networks = [
        _generate_network(config, condition)
        for condition in config["network"]["conditions"]
    ]
    units = [
        {
            "condition_id": network.condition_id,
            "measured_mu": network.measured_mu,
            "strategy": strategy,
            "unit_id": f"{network.condition_id}--{strategy}",
            "logical_decisions": 10,
            "maximum_physical_requests": 20,
        }
        for network in networks
        for strategy in STRATEGIES
    ]
    interval = float(config["llm"]["request_interval_seconds"])
    timeout = float(config["llm"]["timeout_seconds"])
    first_attempts = len(units) * 10
    maximum = len(units) * 20
    # Each independent unit has no interval wait before its first request.
    first_interval_wait = len(units) * 9 * interval
    worst_interval_wait = len(units) * 19 * interval
    return {
        "backend": backend,
        "config_sha256": _sha256(config_path),
        "experiment_id": config["experiment_id"],
        "experiment_units": len(units),
        "logical_decisions": first_attempts,
        "maximum_physical_requests_per_unit": 20,
        "maximum_physical_requests_total": maximum,
        "output_path": str(output),
        "remote_confirmation_required": backend == "groq",
        "request_interval_seconds": interval,
        "resume_requested": bool(resume),
        "runtime_estimate": {
            "first_attempt_interval_only_seconds": first_interval_wait,
            "worst_case_interval_only_seconds": worst_interval_wait,
            "worst_case_with_every_attempt_reaching_timeout_seconds": (
                worst_interval_wait + maximum * timeout
            ),
            "notice": "Conservative bound, not a promise of provider latency.",
        },
        "units": units,
    }


class JointCandidateGenerator(CandidateGenerator):
    """Stable feed candidates: followed authors first, then global posts."""

    def generate(
        self,
        connection: sqlite3.Connection,
        user_id: int,
        recommendation_count: int,
        following_count: int,
    ) -> list[Candidate]:
        followed = {
            int(row[0])
            for row in connection.execute(
                "SELECT followee_id FROM follow WHERE follower_id = ?", (user_id,)
            ).fetchall()
        }
        rows = connection.execute(
            "SELECT post_id, user_id FROM post WHERE user_id != ? ORDER BY post_id DESC",
            (user_id,),
        ).fetchall()
        ordered = sorted(
            ((int(post_id), int(author)) for post_id, author in rows),
            key=lambda item: (item[1] not in followed, -item[0]),
        )
        limit = max(1, int(recommendation_count) + int(following_count))
        return [
            Candidate(
                post_id=post_id,
                base_score=1.0 / (index + 1),
                source="following" if author in followed else "recommendation",
            )
            for index, (post_id, author) in enumerate(ordered[:limit])
        ]


class PlatformActionAdapter:
    def __init__(self, platform: AdaptiveDiffusionPlatform, user_id: int) -> None:
        self.platform = platform
        self.user_id = user_id

    async def refresh(self) -> dict[str, Any]:
        return await self.platform.refresh(self.user_id)

    async def repost(self, post_id: int) -> dict[str, Any]:
        return await self.platform.repost(self.user_id, post_id)

    async def quote_post(self, post_id: int, text: str) -> dict[str, Any]:
        return await self.platform.quote_post(self.user_id, (post_id, text))

    async def report_post(self, post_id: int, text: str) -> dict[str, Any]:
        return await self.platform.report_post(self.user_id, (post_id, text))

    async def do_nothing(self) -> dict[str, Any]:
        return await self.platform.do_nothing(self.user_id)


class FakeJointModel:
    """Deterministic strict-output backend for engineering validation only."""

    def __init__(
        self,
        cache_path: Path,
        *,
        max_attempts: int = 20,
        inject_first_validation_failure: bool = True,
        always_fail_validation: bool = False,
    ) -> None:
        settings = LLMSettings(
            enabled=False,
            provider="groq",
            llm_model="fake-structured-joint-v1",
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
            model_config={"temperature": 0.0, "max_tokens": 512},
            sleeper=lambda _: None,
        )
        self.max_attempts = max_attempts
        self.local_attempts = 0
        self.successful_responses = 0
        self.inject_first_validation_failure = inject_first_validation_failure
        self.always_fail_validation = always_fail_validation
        self._failure_injected = False

    @staticmethod
    def _response_format_choices(response_format: Mapping[str, Any]) -> list[str]:
        return list(
            response_format["json_schema"]["schema"]["properties"]["choice_id"]["enum"]
        )

    @staticmethod
    def _choose(messages: list[dict[str, Any]], choices: list[str]) -> tuple[str, str]:
        user_payload = json.loads(str(messages[1]["content"]))
        profile = str(user_payload["profile"]).lower()
        posts = {
            int(post["post_id"]): str(post.get("content", "")).lower()
            for post in user_payload.get("visible_feed", [])
        }

        def targets(action: str, predicate: Callable[[str], bool]) -> list[str]:
            result = []
            for choice in choices:
                raw_action, separator, raw_post = choice.partition(":")
                if separator and raw_action == action and predicate(posts.get(int(raw_post), "")):
                    result.append(choice)
            return sorted(result, key=lambda value: int(value.partition(":")[2]))

        high = lambda text: "high-risk" in text or "unverified synthetic" in text
        low = lambda text: "verified synthetic benign" in text
        preferences: list[list[str]]
        if "verifier" in profile:
            preferences = [targets("report", high), targets("quote", high)]
        elif "sharing clearly verified" in profile:
            preferences = [targets("repost", low), targets("quote", low)]
        elif "corrective context" in profile:
            preferences = [targets("quote", high), targets("report", high)]
        elif "cautious" in profile:
            preferences = [["ignore"]]
        else:
            preferences = [targets("report", high), targets("repost", low)]
        for options in preferences:
            if options:
                choice = options[0]
                action = choice.partition(":")[0]
                rationale = {
                    "report": "Synthetic audit: the visible claim is explicitly unverified and high risk.",
                    "repost": "Synthetic audit: the visible update is explicitly verified and benign.",
                    "quote": "Synthetic audit: quote with concise corrective context for the visible claim.",
                    "ignore": "Synthetic audit: no platform action selected for this feed.",
                }[action]
                return choice, rationale
        if "ignore" in choices:
            return "ignore", "Synthetic audit: no suitable legal platform action is available."
        choice = choices[0]
        return choice, "Synthetic audit: selected the first remaining legal action."

    async def structured_arun(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> dict[str, Any]:
        del cache_read, defer_unrecovered_error
        if self.local_attempts >= self.max_attempts:
            raise RuntimeError("fake backend physical-attempt cap reached")
        self.local_attempts += 1
        correction = bool(messages and messages[-1].get("content") == CORRECTION_PROMPT)
        should_fail = self.always_fail_validation or (
            self.inject_first_validation_failure
            and not self._failure_injected
            and not correction
        )
        if should_fail:
            self._failure_injected = True
            raise RedactedProviderError(
                "fake strict validation rejection", 400, "json_validate_failed"
            )
        choices = self._response_format_choices(response_format)
        choice, rationale = self._choose(messages, choices)
        self.successful_responses += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"choice_id": choice, "rationale": rationale},
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
            model_config={"temperature": 0.0, "max_tokens": 512},
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
            model_config={"temperature": 0.0, "max_tokens": 512},
        )


def _reset_random_state(seed: int) -> dict[str, Any]:
    random.seed(seed)
    numpy_reset = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_reset = True
    except ImportError:
        pass
    return {
        "python_random_seed": seed,
        "oasis_module_random_seed": seed,
        "project_policy_seed": seed,
        "numpy_seed_reset": numpy_reset,
    }


def _policy_for(
    strategy: StrategyName, config: Mapping[str, Any], network: NetworkSpec
) -> tuple[StaticCOSREFPolicy, dict[str, Any]]:
    exposure = config["exposure"]
    if strategy == "no_intervention":
        pair = (1.0, 1.0)
        direction = "none"
        policy: StaticCOSREFPolicy = NoInterventionPolicy(network.seed)
    elif strategy == "static_cosref":
        static = exposure["static_cosref"]
        pair = (float(static["oasis_keep_intra"]), float(static["oasis_keep_inter"]))
        direction = "symmetric"
        policy = StaticCOSREFPolicy(*pair, network.seed)
    else:
        allocation = allocate_strict_oasis_keep(
            mu=network.measured_mu,
            project_budget=float(exposure["project_l1_budget"]),
            keep_grid=[tuple(map(float, pair)) for pair in exposure["theory_candidates"]],
            tolerance=float(exposure["direction_tolerance"]),
            minimum_strict_gap=float(exposure["minimum_strict_gap"]),
        )
        direction = allocation.direction
        pair = tuple(
            map(float, exposure["theory_selected_by_direction"][direction])
        )
        if pair not in allocation.eligible_oasis_keep_pairs:
            raise JointConfigurationError("selected theory pair is not eligible")
        policy = StaticCOSREFPolicy(*pair, network.seed)
    return policy, {
        "strategy": strategy,
        "measured_mu_used_for_allocation": network.measured_mu,
        "allocation_direction": direction,
        "oasis_keep_intra": pair[0],
        "oasis_keep_inter": pair[1],
        "parameter_l1_cost": project_l1_cost(*pair),
        "paper_omega_intra": None,
        "paper_omega_inter": None,
        "semantic_notice": "OASIS exposure keep probabilities; not paper omega.",
    }


async def _initialize_platform(
    config: Mapping[str, Any],
    network: NetworkSpec,
    strategy: StrategyName,
    database: Path,
) -> tuple[AdaptiveDiffusionPlatform, dict[str, int], dict[int, float], dict[str, Any]]:
    from oasis.social_platform.typing import RecsysType

    from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

    reset = _reset_random_state(network.seed)
    policy, policy_record = _policy_for(strategy, config, network)
    simulation = config["simulation"]
    run_id = f"joint-{network.condition_id}-{strategy}-{network.seed}"
    platform = AdaptiveDiffusionPlatform(
        str(database),
        user_communities=network.communities,
        post_risk_scores={},
        policy=policy,
        random_seed=network.seed,
        candidate_generator=JointCandidateGenerator(),
        run_id=run_id,
        intervention_budget=2.0,
        recsys_type=RecsysType.RANDOM,
        refresh_rec_post_count=int(simulation["recommendation_count"]),
        following_post_count=int(simulation["following_count"]),
        max_rec_post_len=int(simulation["recommendation_buffer"]),
    )
    for user_id, community in sorted(network.communities.items()):
        platform.db.execute(
            """
            INSERT INTO user
            (user_id, agent_id, user_name, name, bio, created_at,
             num_followings, num_followers)
            VALUES (?, ?, ?, ?, ?, 0, 0, 0)
            """,
            (
                user_id,
                user_id,
                f"joint-user-{user_id}",
                f"Synthetic Joint User {user_id}",
                f"synthetic profile in {community}",
            ),
        )
    platform.db.executemany(
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, 0)",
        network.directed_arcs,
    )
    platform.db.commit()
    logical_to_post: dict[str, int] = {}
    root_risk: dict[int, float] = {}
    for post in simulation["initial_posts"]:
        logical = str(post["logical_id"])
        kind = str(post["kind"])
        author = int(post["author"])
        if kind == "root":
            result = await platform.create_post(author, str(post["content"]))
        elif kind == "quote":
            result = await platform.quote_post(
                author,
                (logical_to_post[str(post["parent"])], str(post["content"])),
            )
        elif kind == "repost":
            result = await platform.repost(
                author, logical_to_post[str(post["parent"])]
            )
        else:
            raise JointConfigurationError(f"unsupported initial post kind: {kind}")
        if not result.get("success"):
            raise RuntimeError(f"failed to create initial post {logical}")
        actual = int(result["post_id"])
        logical_to_post[logical] = actual
        if kind == "root":
            risk = float(post["risk_score"])
            root_risk[actual] = risk
            platform.post_risk_scores[actual] = risk
    if simulation["initial_action_history"]:
        raise JointConfigurationError("v1 preregisters an empty initial action history")
    return platform, logical_to_post, root_risk, {
        "run_id": run_id,
        "random_state_reset": reset,
        "policy": policy_record,
    }


def _latest_trace(platform: AdaptiveDiffusionPlatform) -> int:
    return int(
        platform.db.execute("SELECT COALESCE(MAX(rowid), 0) FROM trace").fetchone()[0]
    )


def _trace_after(
    platform: AdaptiveDiffusionPlatform, before: int, user_id: int, choice: str
) -> int | None:
    action = ACTION_TRACE_NAMES[choice.partition(":")[0]]
    row = platform.db.execute(
        "SELECT rowid FROM trace WHERE rowid > ? AND user_id = ? AND action = ? "
        "ORDER BY rowid LIMIT 1",
        (before, user_id, action),
    ).fetchone()
    return None if row is None else int(row[0])


def _history_before(platform: AdaptiveDiffusionPlatform, user_id: int) -> list[str]:
    rows = platform.db.execute(
        "SELECT rowid, action FROM trace WHERE user_id = ? ORDER BY rowid LIMIT 50",
        (user_id,),
    ).fetchall()
    return [f"trace-{int(rowid)}:{action}" for rowid, action in rows]


def _post_context(
    platform: AdaptiveDiffusionPlatform, user_id: int, choice: str
) -> dict[str, Any]:
    action, separator, raw_post = choice.partition(":")
    if not separator:
        return {
            "target_post_id": None,
            "root_post_id": None,
            "root_risk_score": None,
            "community_relation": None,
        }
    post_id = int(raw_post)
    root_id, root_author = platform._root_post(post_id)
    risk = float(platform.post_risk_scores[root_id])
    relation = (
        "intra"
        if platform.user_communities.get(user_id)
        == platform.user_communities[root_author]
        else "inter"
    )
    return {
        "target_post_id": post_id,
        "root_post_id": root_id,
        "root_risk_score": risk,
        "community_relation": relation,
        "action": action,
    }


def _root_adopters(platform: AdaptiveDiffusionPlatform) -> tuple[dict[int, set[int]], dict[int, str]]:
    adopters: dict[int, set[int]] = defaultdict(set)
    root_community: dict[int, str] = {}
    for post_id, author in platform.db.execute(
        "SELECT post_id, user_id FROM post ORDER BY post_id"
    ).fetchall():
        root, root_author = platform._root_post(int(post_id))
        adopters[root].add(int(author))
        root_community[root] = platform.user_communities[root_author]
    return dict(adopters), root_community


def _threshold_shadow(
    *,
    platform: AdaptiveDiffusionPlatform,
    network: NetworkSpec,
    config: Mapping[str, Any],
    snapshot: DecisionSnapshot,
) -> tuple[list[dict[str, Any]], bool]:
    """Calculate pressure only; never refresh, dispatch, mutate, or request."""
    trace_before = _latest_trace(platform)
    impressions_before = int(
        platform.db.execute("SELECT COUNT(*) FROM diffusionguard_impression").fetchone()[0]
    )
    adopters, root_communities = _root_adopters(platform)
    shadow = config["threshold_shadow"]
    engine = ThresholdResponseEngine(
        communities=network.communities,
        contact_edges=network.contacts,
        initial_adopters=adopters,
        threshold=float(shadow["threshold"]),
        paper_omega_intra=float(shadow["paper_omega_intra"]),
        paper_omega_inter=float(shadow["paper_omega_inter"]),
        exposure_gate_enabled=True,
    )
    engine.begin_timestep(snapshot.timestep)
    observable: dict[int, set[int]] = defaultdict(set)
    for post in snapshot.feed_value().get("posts", []):
        root, _ = platform._root_post(int(post["post_id"]))
        observable[root].add(int(post["user_id"]))
    rows: list[dict[str, Any]] = []
    for root in sorted(adopters):
        evaluation = engine.evaluate(
            user_id=snapshot.user_id,
            root_post_id=root,
            root_author_community=root_communities[root],
            observable_adopter_ids=observable[root],
        )
        row = evaluation.as_dict()
        row.update(
            {
                "decision_user_id": snapshot.user_id,
                "provenance": shadow["provenance"],
                "counterfactual_only": True,
                "dispatch_performed": False,
                "remote_requests": 0,
            }
        )
        rows.append(row)
    unchanged = (
        trace_before == _latest_trace(platform)
        and impressions_before
        == int(
            platform.db.execute(
                "SELECT COUNT(*) FROM diffusionguard_impression"
            ).fetchone()[0]
        )
    )
    return rows, unchanged


def _model_for_unit(
    backend: BackendName,
    config: Mapping[str, Any],
    cache_path: Path,
    llm_settings: LLMSettings | None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None],
) -> FakeJointModel | ManagedModelBackend:
    limit = int(config["llm"]["max_physical_requests_per_unit"])
    if backend == "fake":
        return FakeJointModel(
            cache_path,
            max_attempts=limit,
            inject_first_validation_failure=bool(
                config["fake_backend"]["inject_one_json_validate_failed_per_unit"]
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


def _unit_behavior_metrics(
    platform: AdaptiveDiffusionPlatform,
    decisions: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    impressions = platform.db.execute(
        "SELECT user_community, author_community, risk_score, keep_probability, shown "
        "FROM diffusionguard_impression WHERE run_id = ?",
        (platform.run_id,),
    ).fetchall()
    risky = [row for row in impressions if float(row[2]) >= 0.7]
    benign = [row for row in impressions if float(row[2]) <= 0.3]
    succeeded = [row for row in decisions if row["status"] == "succeeded"]
    action_counts = Counter(row["action"] for row in succeeded)
    risk_roots = {
        root for root, risk in platform.post_risk_scores.items() if float(risk) >= 0.7
    }
    risk_posts = 0
    risk_authors: set[int] = set()
    for post_id, author in platform.db.execute(
        "SELECT post_id, user_id FROM post"
    ).fetchall():
        root, _ = platform._root_post(int(post_id))
        if root in risk_roots:
            risk_posts += 1
            risk_authors.add(int(author))
    return {
        "action_counts": {name: int(action_counts.get(name, 0)) for name in ("ignore", "report", "repost", "quote")},
        "candidate_impressions": len(impressions),
        "shown_impressions": sum(bool(row[4]) for row in impressions),
        "high_risk_exposures": sum(bool(row[4]) for row in risky),
        "low_risk_exposures": sum(bool(row[4]) for row in benign),
        "intra_high_risk_exposures": sum(bool(row[4]) and row[0] == row[1] for row in risky),
        "inter_high_risk_exposures": sum(bool(row[4]) and row[0] != row[1] for row in risky),
        "high_risk_reposts": sum(row.get("action") == "repost" and (row.get("root_risk_score") or 0) >= 0.7 for row in succeeded),
        "high_risk_reports": sum(row.get("action") == "report" and (row.get("root_risk_score") or 0) >= 0.7 for row in succeeded),
        "low_risk_reports": sum(row.get("action") == "report" and row.get("root_risk_score") is not None and float(row["root_risk_score"]) <= 0.3 for row in succeeded),
        "risk_cascade_size": risk_posts,
        "risk_community_coverage": len({platform.user_communities[user] for user in risk_authors}) / len(set(platform.user_communities.values())) if risk_authors else 0.0,
        "benign_exposure_loss": sum(not bool(row[4]) for row in benign) / len(benign) if benign else 0.0,
        "parameter_l1_cost": float(policy["parameter_l1_cost"]),
        "realized_intervention_cost": sum(1.0 - float(row[3]) for row in impressions),
        "suppressed_impressions": sum(not bool(row[4]) for row in impressions),
        "threshold_shadow": {
            "evaluations": len(shadow_rows),
            "potential_adoptions": sum(bool(row["should_attempt_adoption"]) for row in shadow_rows),
            "threshold_met_but_exposure_blocked": sum(bool(row["threshold_met_but_exposure_blocked"]) for row in shadow_rows),
            "below_observable_threshold": sum(not bool(row["observable_threshold_satisfied"]) for row in shadow_rows),
            "provenance": "counterfactual shadow diagnostic",
        },
    }


async def _run_unit(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    network: NetworkSpec,
    strategy: StrategyName,
    backend: BackendName,
    directory: Path,
    llm_settings: LLMSettings | None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, directory / "preregistered_config.json")
    _write_json(directory / "network.json", network.public_dict())
    database = directory / "experiment.db"
    platform, logical_posts, root_risk, initialization = await _initialize_platform(
        config, network, strategy, database
    )
    _write_json(directory / "initialization.json", {**initialization, "logical_posts": logical_posts, "root_risk_scores": root_risk})
    _write_json(directory / "exposure_policy.json", initialization["policy"])
    cache_path = directory / "response_cache.sqlite3"
    model = _model_for_unit(
        backend, config, cache_path, llm_settings, model_factory
    )
    store = DecisionSnapshotStore(
        platform.db, now_factory=lambda: FIXED_SNAPSHOT_TIME
    )
    mask_builder = ActionMaskBuilder(platform.db)
    profiles = {int(row["user_id"]): row for row in config["agents"]}
    decisions: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    sequence = 0
    try:
        for timestep, order in enumerate(
            config["simulation"]["active_user_order"], start=1
        ):
            platform.sandbox_clock.time_step = timestep
            for raw_user in order:
                user_id = int(raw_user)
                sequence += 1
                adapter = PlatformActionAdapter(platform, user_id)
                gateway = ActionDecisionGateway(adapter)
                feed = await adapter.refresh()
                gateway.update_visible_feed(feed)
                mask = mask_builder.build(user_id, gateway.visible_post_ids)
                response_format = build_response_format(mask)
                profile = profiles[user_id]
                messages = build_decision_messages(
                    profile=str(profile["description"]), feed=feed, mask=mask
                )
                prompt_text = _canonical(messages)
                prompt_isolated = not any(
                    token in prompt_text
                    for token in (
                        strategy,
                        network.condition_id,
                        "measured_mu",
                        "theory_informed_cosref",
                        "static_cosref",
                        "no_intervention",
                    )
                )
                snapshot = DecisionSnapshot.capture(
                    user_id=user_id,
                    timestep=timestep,
                    feed=feed,
                    visible_post_ids=gateway.visible_post_ids,
                    mask=mask,
                    response_format=response_format,
                    messages=messages,
                    state_identifier=mask_builder.state_identifier(
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
                    behavior_history=_history_before(platform, user_id),
                    platform_notice="Synthetic joint-experiment context.",
                )
                local_shadow, shadow_unchanged = _threshold_shadow(
                    platform=platform,
                    network=network,
                    config=config,
                    snapshot=snapshot,
                )
                for row in local_shadow:
                    row["decision_id"] = pending.decision_id
                shadow_rows.extend(local_shadow)
                attempt: StructuredAttemptResult | None = None
                request_messages = messages
                request_format = response_format
                snapshot_completed = False
                try:
                    attempt = await request_with_structured_correction(model, snapshot)
                    request_messages = attempt.request.messages
                    request_format = attempt.request.response_format
                    structured = parse_structured_response(attempt.response)
                    current_mask = revalidate_snapshot_choice(
                        snapshot, structured.choice_id, mask_builder
                    )
                    context = _post_context(platform, user_id, structured.choice_id)
                    before = _latest_trace(platform)
                    await gateway.dispatch_choice(
                        structured.choice_id,
                        structured.rationale,
                        current_mask.choices,
                    )
                    trace_rowid = _trace_after(
                        platform, before, user_id, structured.choice_id
                    )
                    if trace_rowid is None:
                        raise RuntimeError("dispatcher succeeded without a trace")
                    store.mark_succeeded(
                        pending.decision_id,
                        selected_choice_id=structured.choice_id,
                        rationale=structured.rationale,
                        action_trace_rowid=trace_rowid,
                    )
                    snapshot_completed = True
                    model.runtime.record_structured_completion(retried=attempt.retried)
                    model.cache_structured_response(
                        attempt.response,
                        messages=request_messages,
                        response_format=request_format,
                    )
                    action = structured.choice_id.partition(":")[0]
                    decisions.append(
                        {
                            "action": action,
                            "action_trace_rowid": trace_rowid,
                            "backend": backend,
                            "community_relation": context["community_relation"],
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "legal_choice_count": len(snapshot.legal_choice_ids),
                            "legal_choice_valid": True,
                            "model_prompt_isolated_from_treatment": prompt_isolated,
                            "profile_id": str(profile["profile_id"]),
                            "rationale_sha256": hashlib.sha256(structured.rationale.encode()).hexdigest(),
                            "rationale_training_eligible": False,
                            "root_post_id": context["root_post_id"],
                            "root_risk_score": context["root_risk_score"],
                            "run_id": platform.run_id,
                            "schema_valid": True,
                            "selected_choice_id": structured.choice_id,
                            "status": "succeeded",
                            "strategy": strategy,
                            "structured_retry_used": attempt.retried,
                            "target_post_id": context["target_post_id"],
                            "threshold_shadow_non_mutating": shadow_unchanged,
                            "timestep": timestep,
                            "training_eligible": False,
                            "user_id": user_id,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - experiment boundary
                    model.discard_structured_response(
                        messages=request_messages, response_format=request_format
                    )
                    if not snapshot_completed:
                        status = platform.db.execute(
                            "SELECT status FROM diffusionguard_decision_snapshot WHERE decision_id = ?",
                            (pending.decision_id,),
                        ).fetchone()
                        if status is not None and status[0] == "pending":
                            store.mark_failed(
                                pending.decision_id,
                                "structured_or_dispatch_failure",
                            )
                    decisions.append(
                        {
                            "backend": backend,
                            "condition_id": network.condition_id,
                            "decision_id": pending.decision_id,
                            "failure_category": type(exc).__name__,
                            "feed_post_ids": list(snapshot.visible_post_ids),
                            "model_prompt_isolated_from_treatment": prompt_isolated,
                            "profile_id": str(profile["profile_id"]),
                            "rationale_training_eligible": False,
                            "run_id": platform.run_id,
                            "schema_valid": False,
                            "status": "failed",
                            "strategy": strategy,
                            "structured_retry_used": bool(attempt and attempt.retried),
                            "threshold_shadow_non_mutating": shadow_unchanged,
                            "timestep": timestep,
                            "training_eligible": False,
                            "user_id": user_id,
                        }
                    )
        runtime = model.runtime.stats.to_dict()
        if isinstance(model, FakeJointModel):
            runtime.update(
                {
                    "physical_backend_attempts": model.local_attempts,
                    "physical_remote_attempts": 0,
                    "successful_backend_responses": model.successful_responses,
                    "successful_provider_responses": 0,
                }
            )
        else:
            runtime["physical_backend_attempts"] = runtime["physical_remote_attempts"]
            runtime["successful_backend_responses"] = runtime[
                "successful_provider_responses"
            ]
        policy = initialization["policy"]
        behavior = _unit_behavior_metrics(platform, decisions, shadow_rows, policy)
        counts = store.status_counts()
        summary = {
            "backend": backend,
            "behavior": behavior,
            "completed_decisions": counts["succeeded"],
            "condition_id": network.condition_id,
            "decision_snapshot_status_counts": counts,
            "duration_seconds": time.monotonic() - started,
            "logical_decisions": len(decisions),
            "measured_mu": network.measured_mu,
            "oracle_synthetic_risk_labels": True,
            "policy": policy,
            "run_id": platform.run_id,
            "runtime": runtime,
            "status": "success" if counts["succeeded"] == 10 else "degraded",
            "strategy": strategy,
            "teacher_samples_created": 0,
            "threshold_shadow_mode": "counterfactual_non_dispatching",
        }
        decisions.sort(key=lambda row: str(row["decision_id"]))
        shadow_rows.sort(
            key=lambda row: (str(row["decision_id"]), int(row["root_post_id"]))
        )
        _write_jsonl(directory / "per_decision.jsonl", decisions)
        _write_jsonl(directory / "threshold_shadow.jsonl", shadow_rows)
        _write_json(directory / "runtime_metrics.json", runtime)
        _write_json(directory / "unit_summary.json", summary)
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
    decisions: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    files = (
        "experiment.db",
        "per_decision.jsonl",
        "threshold_shadow.jsonl",
        "runtime_metrics.json",
        "unit_summary.json",
        "response_cache.sqlite3",
        "network.json",
        "exposure_policy.json",
    )
    with sqlite3.connect(directory / "experiment.db") as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        snapshot_count, succeeded = connection.execute(
            "SELECT COUNT(*), SUM(status = 'succeeded') FROM diffusionguard_decision_snapshot"
        ).fetchone()
    return {
        "actual_completed_decisions": int(summary["completed_decisions"]),
        "backend": backend,
        "config_sha256": config_sha256,
        "database_integrity": str(integrity),
        "expected_logical_decisions": 10,
        "file_sha256": {name: _sha256(directory / name) for name in files},
        "implementation_sha256": _implementation_sha(),
        "snapshot_count": int(snapshot_count),
        "succeeded_snapshot_count": int(succeeded or 0),
        "status": str(summary["status"]),
        "unit_id": unit_id,
        "unique_decision_ids": len({str(row["decision_id"]) for row in decisions}),
    }


def _valid_complete(
    directory: Path, *, unit_id: str, backend: BackendName, config_sha256: str
) -> bool:
    try:
        complete = _json(directory / "complete.json")
        if any(
            (
                complete.get("unit_id") != unit_id,
                complete.get("backend") != backend,
                complete.get("config_sha256") != config_sha256,
                complete.get("implementation_sha256") != _implementation_sha(),
                complete.get("expected_logical_decisions") != 10,
                complete.get("actual_completed_decisions") != 10,
                complete.get("snapshot_count") != 10,
                complete.get("succeeded_snapshot_count") != 10,
                complete.get("unique_decision_ids") != 10,
                complete.get("database_integrity") != "ok",
            )
        ):
            return False
        hashes = complete.get("file_sha256", {})
        if not isinstance(hashes, dict) or not hashes:
            return False
        if any(
            not (directory / name).is_file()
            or _sha256(directory / name) != digest
            for name, digest in hashes.items()
        ):
            return False
        with sqlite3.connect(directory / "experiment.db") as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return False


def _quarantine(output: Path, directory: Path, unit_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = output / "quarantine" / f"{unit_id}-{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(target))
    return target


def _audit_outputs(
    output: Path,
    unit_records: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshots = succeeded = bound = feed_consistent = policy_consistent = 0
    leakage: list[str] = []
    duplicate_ids = len(decisions) - len({str(row["decision_id"]) for row in decisions})
    all_action_trace_rows = 0
    pairing_checks: list[bool] = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in unit_records:
        by_condition[str(unit["condition_id"])].append(unit)
    for units in by_condition.values():
        network_hashes: set[str] = set()
        logical_posts: set[str] = set()
        root_risks: set[str] = set()
        random_resets: set[str] = set()
        for unit in units:
            directory = output / unit["relative_directory"]
            network_hashes.add(_sha256(directory / "network.json"))
            initialization = _json(directory / "initialization.json")
            logical_posts.add(_canonical(initialization["logical_posts"]))
            root_risks.add(_canonical(initialization["root_risk_scores"]))
            random_resets.add(_canonical(initialization["random_state_reset"]))
        pairing_checks.append(
            len(units) == 3
            and len(network_hashes) == 1
            and len(logical_posts) == 1
            and len(root_risks) == 1
            and len(random_resets) == 1
        )
    for unit in unit_records:
        database = output / unit["relative_directory"] / "experiment.db"
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM diffusionguard_decision_snapshot ORDER BY decision_sequence"
            ).fetchall()
            snapshots += len(rows)
            run_id = str(unit["run_id"])
            for row in rows:
                if row["status"] == "succeeded":
                    succeeded += 1
                    if row["action_trace_rowid"] is not None:
                        bound += 1
                feed = json.loads(row["feed_json"])
                visible = sorted(json.loads(row["visible_post_ids_json"]))
                feed_ids = sorted(int(post["post_id"]) for post in feed.get("posts", []))
                feed_consistent += feed_ids == visible
                shown = sorted(
                    int(value[0])
                    for value in connection.execute(
                        "SELECT post_id FROM diffusionguard_impression "
                        "WHERE run_id = ? AND user_id = ? AND timestep = ? AND shown = 1",
                        (run_id, int(row["user_id"]), int(row["timestep"])),
                    ).fetchall()
                )
                policy_consistent += shown == visible
                if row["action_trace_rowid"] is not None:
                    for token in json.loads(row["behavior_history_json"]):
                        trace_id = int(str(token).split(":", 1)[0].removeprefix("trace-"))
                        if trace_id >= int(row["action_trace_rowid"]):
                            leakage.append(str(row["decision_id"]))
            placeholders = ",".join("?" for _ in ACTION_TRACE_NAMES.values())
            all_action_trace_rows += int(
                connection.execute(
                    f"SELECT COUNT(*) FROM trace WHERE action IN ({placeholders})",
                    tuple(ACTION_TRACE_NAMES.values()),
                ).fetchone()[0]
            )
    scan_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "integrity_audit.json"
    ]
    secret_free = not any(SECRET_PATTERN.search(path.read_bytes()) for path in scan_files)
    cache_paths = [str(unit["cache_path"]) for unit in unit_records]
    completed = sum(row["status"] == "succeeded" for row in decisions)
    return {
        "action_trace_count": bound,
        "all_action_trace_rows_including_initialization": all_action_trace_rows,
        "cache_isolation_passed": len(cache_paths) == len(set(cache_paths)) == 9,
        "decision_snapshot_count": snapshots,
        "duplicate_decision_id_count": duplicate_ids,
        "failed_samples_in_training_data": 0,
        "feed_consistency_rate": feed_consistent / snapshots if snapshots else 0.0,
        "initial_condition_pairing_passed": all(pairing_checks)
        and len(pairing_checks) == 3,
        "label_leakage_check_passed": not leakage and all(row.get("model_prompt_isolated_from_treatment", False) for row in decisions),
        "leakage_decision_ids": sorted(set(leakage)),
        "policy_impression_consistency_rate": policy_consistent / snapshots if snapshots else 0.0,
        "rationale_training_eligible_count": 0,
        "secret_scan_passed": secret_free,
        "succeeded_snapshot_count": succeeded,
        "teacher_or_training_samples_created": 0,
        "threshold_shadow_non_mutating": all(row.get("threshold_shadow_non_mutating", False) for row in decisions),
        "treatment_isolation_passed": all(pairing_checks)
        and all(row.get("model_prompt_isolated_from_treatment", False) for row in decisions)
        and all(row.get("threshold_shadow_non_mutating", False) for row in decisions),
        "trace_binding_rate": bound / completed if completed else 0.0,
        "unique_decision_id_count": len({str(row["decision_id"]) for row in decisions}),
    }


def _aggregate_behavior(unit_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = defaultdict(dict)
    total_actions: Counter[str] = Counter()
    for unit in unit_summaries:
        behavior = dict(unit["behavior"])
        by_condition[str(unit["condition_id"])][str(unit["strategy"])] = behavior
        total_actions.update(behavior["action_counts"])
    return {
        "action_counts": dict(sorted(total_actions.items())),
        "by_condition_and_strategy": {
            condition: dict(sorted(strategies.items()))
            for condition, strategies in sorted(by_condition.items())
        },
        "fake_backend_notice": "Engineering validation only; not a real LLM effect estimate.",
        "oracle_synthetic_risk_labels": True,
    }


def _paired_comparisons(behavior: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        "high_risk_exposures",
        "low_risk_exposures",
        "intra_high_risk_exposures",
        "inter_high_risk_exposures",
        "high_risk_reposts",
        "high_risk_reports",
        "low_risk_reports",
        "risk_cascade_size",
        "risk_community_coverage",
        "benign_exposure_loss",
        "realized_intervention_cost",
    )
    comparisons = (
        ("theory_informed_cosref", "static_cosref"),
        ("theory_informed_cosref", "no_intervention"),
        ("static_cosref", "no_intervention"),
    )
    result: dict[str, Any] = {}
    for condition, strategies in behavior["by_condition_and_strategy"].items():
        result[condition] = {}
        for treatment, comparator in comparisons:
            result[condition][f"{treatment}_minus_{comparator}"] = {
                metric: float(strategies[treatment][metric])
                - float(strategies[comparator][metric])
                for metric in metrics
            }
    return {
        "comparisons": result,
        "pairing_notice": (
            "Strategies share initial network, profiles, posts, histories, activity "
            "order, random seed, model interface, and prompt template. State may "
            "diverge after treatment; this is initial-condition pairing, not "
            "event-by-event pairing."
        ),
        "statistical_notice": "Fake-backend pipeline values are not real LLM effects.",
    }


def _expected_exposure_differences(behavior: Mapping[str, Any]) -> bool:
    rows = behavior["by_condition_and_strategy"]
    for condition in ("strong-community", "weak-community"):
        values = rows[condition]
        if values["no_intervention"]["high_risk_exposures"] == values["static_cosref"]["high_risk_exposures"]:
            return False
        if values["theory_informed_cosref"]["high_risk_exposures"] == values["static_cosref"]["high_risk_exposures"]:
            return False
    moderate = rows["moderate-mixing"]
    return (
        moderate["theory_informed_cosref"]["parameter_l1_cost"]
        == moderate["static_cosref"]["parameter_l1_cost"]
    )


def _report(
    reliability: Mapping[str, Any],
    integrity: Mapping[str, Any],
    behavior: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    return (
        "# 受限 COSREF + LLM 联合实验框架 v1\n\n"
        "本报告是 Fake Backend 工程验证，不是 Groq 或真实人类行为结果。\n\n"
        "- threshold-response：只读 counterfactual shadow diagnostic；不 dispatch、"
        "不改变 Feed/Prompt、不产生 repost。\n"
        "- 曝光策略：只通过 OASIS Feed 改变结构化动作模型输入。\n"
        "- paper omega 与 OASIS keep probability 分离。\n"
        f"- 完成决策：{reliability['completed_decisions']}/"
        f"{reliability['logical_decisions']}\n"
        f"- 完整单元：{reliability['complete_units']}/9\n"
        f"- Fake backend attempts：{reliability['physical_backend_attempts']}\n"
        f"- 远程请求：{reliability['physical_remote_attempts']}\n"
        f"- Snapshot/trace 绑定率：{integrity['trace_binding_rate']:.3f}\n"
        f"- Feed一致率：{integrity['feed_consistency_rate']:.3f}\n"
        f"- 凭据扫描：{str(integrity['secret_scan_passed']).lower()}\n"
        f"- oracle_synthetic_risk_labels：true\n"
        f"- 真实计划：{plan['logical_decisions']} decisions，最多 "
        f"{plan['maximum_physical_requests_total']} requests。\n\n"
        "正常内容损失使用已知合成风险真值，不能解释为真实部署性能。"
        "动作和传播数值仅验证指标管线。\n"
    )


async def run_joint(
    config_path: Path,
    backend: BackendName,
    output: Path,
    *,
    resume: bool = False,
    llm_settings: LLMSettings | None = None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None] = create_llm_model,
) -> dict[str, Any]:
    config = _load_config(config_path)
    plan = build_plan(config_path, backend, output, resume=resume)
    config_sha = str(plan["config_sha256"])
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    if resume and output.exists() and (output / "manifest.json").is_file():
        manifest = _json(output / "manifest.json")
        if manifest.get("config_sha256") != config_sha:
            raise ResumeIntegrityError("resume refused: configuration SHA-256 changed")
        if manifest.get("backend") != backend:
            raise ResumeIntegrityError("resume refused: backend changed")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "preregistered_config.json").exists():
        shutil.copyfile(config_path, output / "preregistered_config.json")
    networks = {
        spec.condition_id: spec
        for spec in (
            _generate_network(config, condition)
            for condition in config["network"]["conditions"]
        )
    }
    unit_records: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    resumed_units = 0
    started = time.monotonic()
    for condition in config["network"]["conditions"]:
        condition_id = str(condition["condition_id"])
        network = networks[condition_id]
        for strategy_value in STRATEGIES:
            strategy: StrategyName = strategy_value
            unit_id = f"{condition_id}--{strategy}"
            relative = Path("units") / condition_id / strategy
            directory = output / relative
            if resume and directory.exists() and _valid_complete(
                directory,
                unit_id=unit_id,
                backend=backend,
                config_sha256=config_sha,
            ):
                decisions = _jsonl(directory / "per_decision.jsonl")
                summary = _json(directory / "unit_summary.json")
                runtime = _json(directory / "runtime_metrics.json")
                resumed_units += 1
            else:
                if directory.exists():
                    _quarantine(output, directory, unit_id)
                pending = output / ".pending" / unit_id
                if pending.exists():
                    _quarantine(output, pending, f"pending-{unit_id}")
                decisions, summary, runtime = await _run_unit(
                    config=config,
                    config_path=config_path,
                    network=network,
                    strategy=strategy,
                    backend=backend,
                    directory=pending,
                    llm_settings=llm_settings,
                    model_factory=model_factory,
                )
                directory.parent.mkdir(parents=True, exist_ok=True)
                pending.replace(directory)
                _write_json(
                    directory / "complete.json",
                    _complete_payload(
                        directory,
                        unit_id=unit_id,
                        backend=backend,
                        config_sha256=config_sha,
                        decisions=decisions,
                        summary=summary,
                    ),
                )
            all_decisions.extend(decisions)
            summaries.append(summary)
            runtimes.append(runtime)
            unit_records.append(
                {
                    "cache_path": str(relative / "response_cache.sqlite3"),
                    "complete": _valid_complete(
                        directory,
                        unit_id=unit_id,
                        backend=backend,
                        config_sha256=config_sha,
                    ),
                    "condition_id": condition_id,
                    "logical_decisions": len(decisions),
                    "measured_mu": network.measured_mu,
                    "relative_directory": str(relative),
                    "run_id": str(summary["run_id"]),
                    "strategy": strategy,
                    "unit_id": unit_id,
                }
            )
    all_decisions.sort(key=lambda row: str(row["decision_id"]))
    summaries.sort(key=lambda row: (str(row["condition_id"]), str(row["strategy"])))
    behavior = _aggregate_behavior(summaries)
    paired = _paired_comparisons(behavior)
    reliability_totals = Counter()
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
            "unrecovered_error_count",
        ):
            reliability_totals[key] += int(runtime.get(key, 0))
    completed = sum(row["status"] == "succeeded" for row in all_decisions)
    reliability = {
        "backend": backend,
        "cache_hits": reliability_totals["cache_hits"],
        "complete_units": sum(bool(unit["complete"]) for unit in unit_records),
        "completed_decisions": completed,
        "dispatcher_failure_count": len(all_decisions) - completed,
        "dispatcher_success_count": completed,
        "first_attempt_count": reliability_totals["first_attempt_count"],
        "first_attempt_success_count": reliability_totals["first_attempt_success_count"],
        "first_attempt_success_rate": reliability_totals["first_attempt_success_count"] / reliability_totals["first_attempt_count"] if reliability_totals["first_attempt_count"] else 0.0,
        "json_validate_failed_count": reliability_totals["json_validate_failed_count"],
        "legal_choice_count": sum(bool(row.get("legal_choice_valid")) for row in all_decisions),
        "logical_decisions": len(all_decisions),
        "physical_backend_attempts": reliability_totals["physical_backend_attempts"],
        "physical_remote_attempts": reliability_totals["physical_remote_attempts"],
        "resumed_units": resumed_units,
        "schema_valid_count": sum(bool(row.get("schema_valid")) for row in all_decisions),
        "structured_retry_attempt_count": reliability_totals["structured_retry_attempt_count"],
        "structured_retry_failure_count": reliability_totals["structured_retry_failure_count"],
        "structured_retry_success_count": reliability_totals["structured_retry_success_count"],
        "unrecovered_error_count": reliability_totals["unrecovered_error_count"] + len(all_decisions) - completed,
    }
    manifest = {
        "backend": backend,
        "config_sha256": config_sha,
        "experiment_id": config["experiment_id"],
        "implementation_sha256": _implementation_sha(),
        "initial_conditions_paired_not_eventwise": True,
        "logical_decisions": 90,
        "maximum_physical_requests_per_unit": 20,
        "maximum_physical_requests_total": 180,
        "oracle_synthetic_risk_labels": True,
        "remote_api_calls": reliability["physical_remote_attempts"],
        "unit_count": 9,
        "units": unit_records,
    }
    _write_json(output / "manifest.json", manifest)
    _write_jsonl(output / "per_decision.jsonl", all_decisions)
    _write_json(output / "reliability_summary.json", reliability)
    _write_json(output / "behavior_propagation_summary.json", behavior)
    _write_json(output / "paired_comparisons.json", paired)
    integrity = _audit_outputs(output, unit_records, all_decisions)
    exposure_difference = _expected_exposure_differences(behavior)
    integrity["expected_exposure_differences_observed"] = exposure_difference
    integrity["complete_unit_count"] = reliability["complete_units"]
    integrity["cross_strategy_cache_pollution_detected"] = False
    integrity["physical_request_limit_enforced"] = all(
        int(runtime.get("physical_backend_attempts", 0)) <= 20 for runtime in runtimes
    ) and int(reliability["physical_backend_attempts"]) <= 180
    _write_json(output / "integrity_audit.json", integrity)
    ready = all(
        (
            completed == 90,
            reliability["complete_units"] == 9,
            reliability["dispatcher_failure_count"] == 0,
            reliability["unrecovered_error_count"] == 0,
            integrity["decision_snapshot_count"] == 90,
            integrity["succeeded_snapshot_count"] == 90,
            integrity["trace_binding_rate"] == 1.0,
            integrity["feed_consistency_rate"] == 1.0,
            integrity["policy_impression_consistency_rate"] == 1.0,
            integrity["threshold_shadow_non_mutating"],
            integrity["initial_condition_pairing_passed"],
            integrity["treatment_isolation_passed"],
            integrity["cache_isolation_passed"],
            integrity["label_leakage_check_passed"],
            integrity["secret_scan_passed"],
            integrity["physical_request_limit_enforced"],
            exposure_difference,
        )
    )
    status = "success" if ready else ("degraded" if completed else "failed")
    summary = {
        "backend": backend,
        "completed_decisions": completed,
        "duration_seconds": time.monotonic() - started,
        "exit_code": EXIT_CODES[status],
        "fake_validation_ready_for_remote_pilot": bool(ready and backend == "fake"),
        "logical_decisions": len(all_decisions),
        "oracle_synthetic_risk_labels": True,
        "physical_remote_attempts": reliability["physical_remote_attempts"],
        "status": status,
        "teacher_or_training_samples_created": 0,
    }
    _write_json(output / "summary.json", summary)
    _atomic_text(output / "REPORT.md", _report(reliability, integrity, behavior, plan))
    return {
        "behavior": behavior,
        "integrity": integrity,
        "manifest": manifest,
        "paired_comparisons": paired,
        "reliability": reliability,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("fake", "groq"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-remote-run", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plan = build_plan(
            args.config, args.backend, args.output, resume=args.resume
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"joint experiment configuration failed: {exc}") from None
    if args.plan:
        print(_canonical(plan, pretty=True), end="")
        return
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
            run_joint(
                args.config,
                args.backend,
                args.output,
                resume=args.resume,
                llm_settings=settings,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(f"joint experiment failed: {exc}") from None
    print(_canonical(result["summary"], pretty=True), end="")
    raise SystemExit(int(result["summary"]["exit_code"]))


if __name__ == "__main__":
    main()
