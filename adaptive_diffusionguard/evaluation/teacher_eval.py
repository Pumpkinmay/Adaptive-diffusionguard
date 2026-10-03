"""Offline, reproducible 100-decision teacher-quality benchmark."""

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
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from oasis.social_platform.typing import RecsysType

from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy
from adaptive_diffusionguard.llm.action_gateway import ActionDecisionGateway
from adaptive_diffusionguard.llm.model_factory import (
    LLMSettings,
    create_llm_model,
)
from adaptive_diffusionguard.llm.runtime import ManagedModelBackend
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    ActionMaskBuilder,
    DecisionSnapshot,
    StructuredActionResponse,
    build_decision_messages,
    build_response_format,
    parse_structured_response,
)
from adaptive_diffusionguard.llm.structured_retry import (
    StructuredAttemptResult,
    request_with_structured_correction,
)
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform
from adaptive_diffusionguard.recommendation.base import Candidate, CandidateGenerator
from adaptive_diffusionguard.storage.decision_snapshots import DecisionSnapshotStore
from training.dataset_pipeline import build_from_decision_snapshots

from .suite import DecisionRubric, classify_rationale_stance, load_suite

BackendName = Literal["fake", "random_legal", "deterministic_risk_rule", "groq"]
ACTION_TRACE_NAMES = ("repost", "quote_post", "report_post", "do_nothing")
SECRET_PATTERN_BYTES = re.compile(rb"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")
FIXED_SNAPSHOT_TIME = "2026-01-01T00:00:00.000000+00:00"
MAX_PHYSICAL_REQUESTS_PER_BATCH = 30
EXIT_CODES = {"success": 0, "failed": 1, "degraded": 2}


class EvaluationConfigurationError(ValueError):
    """Fail before output creation, environment loading, or model creation."""


class ResumeIntegrityError(RuntimeError):
    """A completed batch cannot be trusted for resumable aggregation."""


def _canonical(value: Any, *, pretty: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs) + "\n"


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(_canonical(row) for row in rows)
    _atomic_text(path, payload)


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _available_batches(suite: dict[str, Any]) -> list[str]:
    return [str(batch["batch_id"]) for batch in suite["batches"]]


def _select_batches(
    suite: dict[str, Any], backend: BackendName, selected_batch: str | None
) -> list[dict[str, Any]]:
    available = _available_batches(suite)
    if backend == "groq" and selected_batch is None:
        raise EvaluationConfigurationError(
            "backend=groq requires exactly one explicit --batch"
        )
    if selected_batch is None:
        return list(suite["batches"])
    if "," in selected_batch or selected_batch not in available:
        raise EvaluationConfigurationError(
            f"unknown batch {selected_batch!r}; choose one of: {', '.join(available)}"
        )
    return [
        batch for batch in suite["batches"] if str(batch["batch_id"]) == selected_batch
    ]


def build_plan(
    suite_path: Path,
    backend: BackendName,
    selected_batch: str | None,
    output: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Read only suite metadata and return a non-sensitive execution plan."""
    suite = load_suite(suite_path)
    selected = _select_batches(suite, backend, selected_batch)
    scenario_ids = [
        str(scenario_id)
        for batch in selected
        for scenario_id in batch["scenarios"]
    ]
    return {
        "backend": backend,
        "logical_decisions": len(scenario_ids) * 5,
        "maximum_physical_requests": (
            MAX_PHYSICAL_REQUESTS_PER_BATCH * len(selected)
        ),
        "output_path": str(output),
        "resume_requested": bool(resume),
        "scenario_ids": scenario_ids,
        "selected_batch": selected_batch,
    }


class FixedCandidateGenerator(CandidateGenerator):
    """Return the scenario's declared candidates in a fixed order."""

    def __init__(self) -> None:
        self.by_user: dict[int, list[int]] = {}

    def generate(
        self,
        connection: sqlite3.Connection,
        user_id: int,
        recommendation_count: int,
        following_count: int,
    ) -> list[Candidate]:
        del connection, recommendation_count, following_count
        return [
            Candidate(post_id=post_id, base_score=1.0 / (index + 1), source="suite")
            for index, post_id in enumerate(self.by_user.get(user_id, []))
        ]


class PlatformActionAdapter:
    """Expose the gateway protocol without constructing an LLM agent."""

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


@dataclass(frozen=True, slots=True)
class ScenarioState:
    platform: AdaptiveDiffusionPlatform
    logical_to_post: dict[str, int]
    post_to_logical: dict[int, str]
    root_logical: dict[str, str]
    risk_by_root: dict[str, float]
    author_by_root: dict[str, int]


def _root_logical(posts: list[dict[str, Any]], logical_id: str) -> str:
    by_id = {str(post["post_id"]): post for post in posts}
    current = logical_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        parent = by_id[current].get("parent_post")
        if parent is None:
            return current
        current = str(parent)
    raise ValueError(f"post relation cycle at {logical_id}")


async def _initialize_scenario(
    scenario: dict[str, Any], profiles: list[dict[str, Any]], db_path: Path
) -> ScenarioState:
    communities = {index: str(value) for index, value in enumerate(scenario["communities"])}
    candidates = FixedCandidateGenerator()
    cosref = scenario["cosref"]
    platform = AdaptiveDiffusionPlatform(
        str(db_path),
        user_communities=communities,
        post_risk_scores={},
        policy=StaticCOSREFPolicy(
            float(cosref["omega_intra"]),
            float(cosref["omega_inter"]),
            int(scenario["random_seed"]),
        ),
        random_seed=int(scenario["random_seed"]),
        candidate_generator=candidates,
        run_id=f"teacher-eval-{scenario['scenario_id']}",
        recsys_type=RecsysType.RANDOM,
        refresh_rec_post_count=20,
        following_post_count=20,
        max_rec_post_len=20,
    )
    platform.sandbox_clock.time_step = 0
    for user_id, profile in enumerate(profiles):
        result = await platform.sign_up(
            user_id,
            (
                f"synthetic_eval_{scenario['scenario_id']}_{user_id}",
                str(profile["profile_id"]),
                str(profile["description"]),
            ),
        )
        if not result.get("success"):
            raise RuntimeError("synthetic agent initialization failed")

    logical_to_post: dict[str, int] = {}
    post_to_logical: dict[int, str] = {}
    posts = list(scenario.get("posts", []))
    roots = {str(post["post_id"]): _root_logical(posts, str(post["post_id"])) for post in posts}
    risk_by_root: dict[str, float] = {}
    author_by_root: dict[str, int] = {}
    for post in posts:
        logical_id = str(post["post_id"])
        author = int(post["author"])
        kind = str(post.get("kind", "root"))
        if kind == "root":
            result = await platform.create_post(author, str(post["content"]))
        elif kind == "repost":
            result = await platform.repost(
                author, logical_to_post[str(post["parent_post"])]
            )
        elif kind == "quote":
            result = await platform.quote_post(
                author,
                (
                    logical_to_post[str(post["parent_post"])],
                    str(post["content"]),
                ),
            )
        else:
            raise ValueError(f"unsupported post kind: {kind}")
        if not result.get("success"):
            raise RuntimeError(f"synthetic post initialization failed: {logical_id}")
        actual_id = int(result["post_id"])
        logical_to_post[logical_id] = actual_id
        post_to_logical[actual_id] = logical_id
        root = roots[logical_id]
        if kind == "root":
            risk_by_root[root] = float(post["risk_score"])
            author_by_root[root] = author
            platform.post_risk_scores[actual_id] = float(post["risk_score"])

    for follower, followee in scenario.get("follows", []):
        result = await platform.follow(int(follower), int(followee))
        if not result.get("success"):
            raise RuntimeError("synthetic follow initialization failed")
    for event in scenario.get("initial_history", []):
        user_id = int(event["user_id"])
        post_id = logical_to_post[str(event["post_id"])]
        action = str(event["action"])
        if action == "report":
            result = await platform.report_post(user_id, (post_id, "synthetic prior report"))
        elif action == "repost":
            result = await platform.repost(user_id, post_id)
        elif action == "quote":
            result = await platform.quote_post(user_id, (post_id, "synthetic prior quote"))
        else:
            raise ValueError(f"unsupported initial action: {action}")
        if not result.get("success"):
            raise RuntimeError("synthetic initial-history action failed")
    candidates.by_user = {
        int(user): [logical_to_post[str(post)] for post in logical_posts]
        for user, logical_posts in scenario["feeds"].items()
    }
    return ScenarioState(
        platform=platform,
        logical_to_post=logical_to_post,
        post_to_logical=post_to_logical,
        root_logical=roots,
        risk_by_root=risk_by_root,
        author_by_root=author_by_root,
    )


def _history_before(platform: AdaptiveDiffusionPlatform, user_id: int) -> list[str]:
    rows = platform.db.execute(
        "SELECT rowid, action FROM trace WHERE user_id = ? ORDER BY rowid DESC LIMIT 20",
        (user_id,),
    ).fetchall()
    return [f"trace-{int(rowid)}:{action}" for rowid, action in reversed(rows)]


def _latest_trace(platform: AdaptiveDiffusionPlatform) -> int:
    return int(platform.db.execute("SELECT COALESCE(MAX(rowid), 0) FROM trace").fetchone()[0])


def _trace_after(
    platform: AdaptiveDiffusionPlatform, after: int, user_id: int, choice: str
) -> int | None:
    expected = {
        "repost": "repost",
        "quote": "quote_post",
        "report": "report_post",
        "ignore": "do_nothing",
    }[choice.partition(":")[0]]
    row = platform.db.execute(
        "SELECT rowid FROM trace WHERE rowid > ? AND user_id = ? AND action = ? "
        "ORDER BY rowid LIMIT 1",
        (after, user_id, expected),
    ).fetchone()
    return None if row is None else int(row[0])


def _choice_for_rule(
    rule: Any, mask: ActionMask, state: ScenarioState
) -> str | None:
    if rule.action == "ignore":
        return "ignore" if "ignore" in mask.choices else None
    for choice in mask.choices:
        action, _, raw_id = choice.partition(":")
        if action != rule.action:
            continue
        logical = state.post_to_logical.get(int(raw_id))
        if logical is not None and state.root_logical[logical] == rule.root_post:
            return choice
    return None


def _rationale(action: str, stance: str = "not_applicable") -> str:
    if action == "quote":
        tag = stance if stance in {"challenge", "correct", "support"} else "unclear"
        return f"[stance:{tag}] Synthetic benchmark rationale."
    if action == "report":
        return "Synthetic rubric indicates that reporting this item is warranted."
    if action == "repost":
        return "Synthetic rubric permits sharing this item."
    return "Synthetic rubric permits no platform action."


def _select_choice(
    backend: BackendName,
    *,
    mask: ActionMask,
    rubric: DecisionRubric,
    state: ScenarioState,
    rng: random.Random,
) -> tuple[str, str]:
    if backend == "fake":
        for rules in (rubric.aligned, rubric.acceptable):
            for rule in rules:
                choice = _choice_for_rule(rule, mask, state)
                if choice is not None:
                    return choice, _rationale(rule.action, rule.stance or "not_applicable")
        return "ignore", _rationale("ignore")
    if backend == "random_legal":
        choice = rng.choice(list(mask.choices))
        return choice, _rationale(choice.partition(":")[0])
    if backend == "deterministic_risk_rule":
        candidates: list[tuple[float, int, str]] = []
        for choice in mask.choices:
            action, _, raw_id = choice.partition(":")
            if action not in {"report", "repost"}:
                continue
            logical = state.post_to_logical[int(raw_id)]
            risk = state.risk_by_root[state.root_logical[logical]]
            priority = 0 if action == "report" else 1
            candidates.append((risk, priority, choice))
        high_reports = sorted(
            (item for item in candidates if item[1] == 0 and item[0] >= 0.7),
            key=lambda item: (-item[0], item[2]),
        )
        if high_reports:
            return high_reports[0][2], _rationale("report")
        low_reposts = sorted(
            (item for item in candidates if item[1] == 1 and item[0] <= 0.3),
            key=lambda item: (item[0], item[2]),
        )
        if low_reposts:
            return low_reposts[0][2], _rationale("repost")
        return "ignore", _rationale("ignore")
    raise RuntimeError(
        "groq backend is intentionally disabled in this offline benchmark build"
    )


def _target_context(
    choice: str, state: ScenarioState, communities: list[str]
) -> dict[str, Any]:
    action, _, raw_id = choice.partition(":")
    if action == "ignore":
        return {
            "target_post_id": None,
            "target_logical_post": None,
            "root_post_id": None,
            "root_logical_post": None,
            "root_risk_score": None,
            "author_community": None,
            "community_relation": None,
        }
    post_id = int(raw_id)
    logical = state.post_to_logical[post_id]
    root = state.root_logical[logical]
    root_actual = state.logical_to_post[root]
    author = state.author_by_root[root]
    return {
        "target_post_id": post_id,
        "target_logical_post": logical,
        "root_post_id": root_actual,
        "root_logical_post": root,
        "root_risk_score": state.risk_by_root[root],
        "author_community": communities[author],
    }


async def _run_scenario(
    scenario: dict[str, Any],
    profiles: list[dict[str, Any]],
    backend: BackendName,
    db_path: Path,
    batch_id: str,
    model: ManagedModelBackend | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = await _initialize_scenario(scenario, profiles, db_path)
    platform = state.platform
    mask_builder = ActionMaskBuilder(platform.db)
    store = DecisionSnapshotStore(platform.db, now_factory=lambda: FIXED_SNAPSHOT_TIME)
    rng = random.Random(
        int(scenario["random_seed"])
        + {"fake": 0, "random_legal": 1, "deterministic_risk_rule": 2, "groq": 3}[
            backend
        ]
    )
    decisions: list[dict[str, Any]] = []
    communities = [str(value) for value in scenario["communities"]]
    try:
        for user_id in range(5):
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
            snapshot = DecisionSnapshot.capture(
                user_id=user_id,
                timestep=0,
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
                decision_sequence=user_id + 1,
                user_profile=str(profile["description"]),
                community=communities[user_id],
                behavior_history=_history_before(platform, user_id),
                platform_notice="Synthetic rule-based teacher evaluation.",
            )
            rubric = DecisionRubric.from_dict(scenario["rubric"][str(user_id)])
            attempt_result: StructuredAttemptResult | None = None
            request_messages = messages
            request_response_format = response_format
            completion_recorded = False
            try:
                if backend == "groq":
                    if model is None:
                        raise RuntimeError("groq backend model was not initialized")
                    attempt_result = await request_with_structured_correction(
                        model, snapshot
                    )
                    request_messages = attempt_result.request.messages
                    request_response_format = attempt_result.request.response_format
                    structured = parse_structured_response(attempt_result.response)
                else:
                    choice, rationale = _select_choice(
                        backend, mask=mask, rubric=rubric, state=state, rng=rng
                    )
                    # Local backends cross the same strict Pydantic boundary as
                    # a remote structured response.
                    structured = StructuredActionResponse.model_validate(
                        {"choice_id": choice, "rationale": rationale}
                    )
                choice, rationale = structured.choice_id, structured.rationale
                if choice not in snapshot.legal_choice_ids:
                    raise ValueError("structured choice is outside the frozen mask")
                before = _latest_trace(platform)
                await gateway.dispatch_choice(choice, rationale, mask.choices)
                trace_rowid = _trace_after(platform, before, user_id, choice)
                if trace_rowid is None:
                    raise RuntimeError("dispatcher succeeded without an OASIS trace")
                store.mark_succeeded(
                    pending.decision_id,
                    selected_choice_id=choice,
                    rationale=rationale,
                    action_trace_rowid=trace_rowid,
                )
                if model is not None and attempt_result is not None:
                    model.runtime.record_structured_completion(
                        retried=attempt_result.retried
                    )
                    completion_recorded = True
                    model.cache_structured_response(
                        attempt_result.response,
                        messages=request_messages,
                        response_format=request_response_format,
                    )
                action = choice.partition(":")[0]
                target = _target_context(choice, state, communities)
                if target["author_community"] is not None:
                    target["community_relation"] = (
                        "intra"
                        if target["author_community"] == communities[user_id]
                        else "inter"
                    )
                stance = classify_rationale_stance(action, rationale)
                quality = (
                    rubric.classify(
                        action=action,
                        root_post=target["root_logical_post"],
                        stance=stance,
                    )
                    if scenario["include_in_quality_score"]
                    else "unscored"
                )
                feed_roots = []
                for post_id in snapshot.visible_post_ids:
                    logical = state.post_to_logical[post_id]
                    root = state.root_logical[logical]
                    feed_roots.append(
                        {
                            "logical_post": logical,
                            "root_logical_post": root,
                            "risk_score": state.risk_by_root[root],
                        }
                    )
                decisions.append(
                    {
                        "action": action,
                        "action_label_eligible": True,
                        "action_trace_rowid": trace_rowid,
                        "backend": backend,
                        "batch_id": batch_id,
                        "community_relation": target["community_relation"],
                        "decision_id": pending.decision_id,
                        "feed_roots": feed_roots,
                        "legal_choice_ids": list(mask.choices),
                        "profile_id": str(profile["profile_id"]),
                        "quality_class": quality,
                        "random_seed": int(scenario["random_seed"]),
                        "rationale": rationale,
                        "rationale_stance": stance,
                        "rationale_training_eligible": False,
                        "rationale_quality_status": "manual_review_required",
                        "run_id": platform.run_id,
                        "scenario_id": str(scenario["scenario_id"]),
                        "scenario_included_in_quality_score": bool(
                            scenario["include_in_quality_score"]
                        ),
                        "selected_choice_id": choice,
                        "status": "succeeded",
                        "structured_retry_used": bool(
                            attempt_result and attempt_result.retried
                        ),
                        "structured_schema_valid": True,
                        "user_community": communities[user_id],
                        "user_id": user_id,
                        **target,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - local benchmark boundary
                if model is not None:
                    if (
                        attempt_result is not None
                        and attempt_result.retried
                        and not completion_recorded
                    ):
                        model.runtime.record_structured_retry_failure()
                    model.discard_structured_response(
                        messages=request_messages,
                        response_format=request_response_format,
                    )
                store.mark_failed(pending.decision_id, "local_backend_or_dispatch_failure")
                decisions.append(
                    {
                        "backend": backend,
                        "batch_id": batch_id,
                        "decision_id": pending.decision_id,
                        "failure_category": type(exc).__name__,
                        "profile_id": str(profile["profile_id"]),
                        "run_id": platform.run_id,
                        "scenario_id": str(scenario["scenario_id"]),
                        "status": "failed",
                        "structured_retry_used": bool(
                            attempt_result and attempt_result.retried
                        ),
                        "structured_schema_valid": False,
                        "user_id": user_id,
                    }
                )
        build = build_from_decision_snapshots(db_path, allow_failed=True)
        snapshot_counts = store.status_counts()
        impression_count = int(
            platform.db.execute(
                "SELECT COUNT(*) FROM diffusionguard_impression"
            ).fetchone()[0]
        )
        summary = {
            "action_counts": dict(sorted(Counter(row.get("action") for row in decisions if row["status"] == "succeeded").items())),
            "backend": backend,
            "cosref": dict(scenario["cosref"]),
            "coverage": list(scenario["coverage"]),
            "decision_snapshot_status_counts": snapshot_counts,
            "decisions": len(decisions),
            "description": str(scenario["description"]),
            "impression_count": impression_count,
            "run_id": platform.run_id,
            "random_seed": int(scenario["random_seed"]),
            "scenario_id": str(scenario["scenario_id"]),
            "scored": bool(scenario["include_in_quality_score"]),
            "succeeded": snapshot_counts["succeeded"],
            "teacher_candidates": len(build.examples),
        }
        return decisions, summary
    finally:
        platform.db.close()


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in rows if row["status"] == "succeeded"]
    counts = Counter(row["quality_class"] for row in succeeded)
    scored_denominator = counts["aligned"] + counts["acceptable"] + counts["concerning"]
    quote_stances = Counter(
        row["rationale_stance"] for row in succeeded if row["action"] == "quote"
    )
    ignore_risk = Counter()
    missed = 0
    report_low = 0
    repost_high = 0
    for row in succeeded:
        if row["action"] == "ignore":
            risks = [float(item["risk_score"]) for item in row["feed_roots"]]
            bucket = "empty" if not risks else (
                "contains_high" if any(risk >= 0.7 for risk in risks) else "low_or_ambiguous_only"
            )
            ignore_risk[bucket] += 1
            if bucket == "contains_high" and any(
                choice.startswith("report:") for choice in row["legal_choice_ids"]
            ):
                missed += 1
        risk = row["root_risk_score"]
        if row["action"] == "report" and risk is not None and risk <= 0.3:
            report_low += 1
        if row["action"] == "repost" and risk is not None and risk >= 0.7:
            repost_high += 1

    def stratify(field: str) -> dict[str, dict[str, int]]:
        result: dict[str, Counter[str]] = defaultdict(Counter)
        for row in succeeded:
            key = row.get(field)
            result["unavailable" if key is None else str(key)][row["quality_class"]] += 1
        return {
            key: {name: int(values.get(name, 0)) for name in ("aligned", "acceptable", "concerning", "unscored")}
            for key, values in sorted(result.items())
        }

    return {
        "acceptable_count": counts["acceptable"],
        "acceptable_rate": counts["acceptable"] / scored_denominator if scored_denominator else None,
        "action_counts": dict(sorted(Counter(row["action"] for row in succeeded).items())),
        "aligned_count": counts["aligned"],
        "aligned_rate": counts["aligned"] / scored_denominator if scored_denominator else None,
        "concerning_count": counts["concerning"],
        "concerning_rate": counts["concerning"] / scored_denominator if scored_denominator else None,
        "ignore_feed_risk_distribution": dict(sorted(ignore_risk.items())),
        "missed_intervention_candidate_count": missed,
        "quote_stance_counts": {name: quote_stances.get(name, 0) for name in ("challenge", "correct", "support", "unclear")},
        "report_low_risk_count": report_low,
        "repost_high_risk_count": repost_high,
        "scored_denominator": scored_denominator,
        "stratified_by_community_relation": stratify("community_relation"),
        "stratified_by_profile": stratify("profile_id"),
        "stratified_by_root_risk": stratify("root_risk_score"),
        "stratified_by_scenario": stratify("scenario_id"),
        "synthetic_rubric_notice": "Rule-based synthetic rubric; not a human-behavior gold standard.",
        "unscored_count": counts["unscored"],
    }


def _reliability_summary(
    rows: list[dict[str, Any]],
    backend: BackendName,
    batch_runtime: list[dict[str, Any]],
) -> dict[str, Any]:
    logical = len(rows)
    completed = sum(row["status"] == "succeeded" for row in rows)
    fake_invocations = logical if backend == "fake" else 0
    totals = Counter()
    for stats in batch_runtime:
        for key in (
            "cache_hits",
            "first_attempt_count",
            "first_attempt_success_count",
            "json_validate_failed_count",
            "physical_remote_attempts",
            "rate_limit_count",
            "structured_retry_attempt_count",
            "structured_retry_failure_count",
            "structured_retry_success_count",
            "successful_provider_responses",
            "unrecovered_error_count",
        ):
            totals[key] += int(stats.get(key, 0))
    remote = backend == "groq"
    first_attempts = totals["first_attempt_count"] if remote else logical
    first_successes = totals["first_attempt_success_count"] if remote else completed
    return {
        "backend": backend,
        "cache_hits": totals["cache_hits"] if remote else 0,
        "completed_decisions": completed,
        "dispatcher_failure_count": logical - completed,
        "dispatcher_success_count": completed,
        "first_attempt_count": first_attempts,
        "first_attempt_success_count": first_successes,
        "first_attempt_success_rate": first_successes / first_attempts if first_attempts else 0.0,
        "invalid_choice_count": 0,
        "json_validate_failed_count": totals["json_validate_failed_count"] if remote else 0,
        "local_backend_invocations": fake_invocations,
        "logical_decisions": logical,
        "physical_remote_attempts": totals["physical_remote_attempts"] if remote else 0,
        "post_retry_success_rate": completed / logical if logical else 0.0,
        "rate_limit_count": totals["rate_limit_count"] if remote else 0,
        "schema_valid_count": sum(
            bool(row.get("structured_schema_valid")) for row in rows
        ),
        "structured_retry_attempt_count": totals["structured_retry_attempt_count"] if remote else 0,
        "structured_retry_failure_count": totals["structured_retry_failure_count"] if remote else 0,
        "structured_retry_success_count": totals["structured_retry_success_count"] if remote else 0,
        "successful_provider_responses": totals["successful_provider_responses"] if remote else fake_invocations,
        "unrecovered_error_count": totals["unrecovered_error_count"] if remote else logical - completed,
    }


def _dataset_validation(
    rows: list[dict[str, Any]], scenario_summaries: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    ids = [row["decision_id"] for row in rows]
    succeeded = [row for row in rows if row["status"] == "succeeded"]
    database_snapshots = 0
    trace_bound = 0
    feed_consistent = 0
    leakage_errors: list[str] = []
    for scenario in scenario_summaries:
        db = output / "batches" / scenario["batch_id"] / "scenarios" / f"{scenario['scenario_id']}.db"
        with sqlite3.connect(db) as connection:
            connection.row_factory = sqlite3.Row
            snapshots = connection.execute(
                "SELECT * FROM diffusionguard_decision_snapshot ORDER BY decision_sequence"
            ).fetchall()
            database_snapshots += len(snapshots)
            for snapshot in snapshots:
                feed = json.loads(snapshot["feed_json"])
                visible = json.loads(snapshot["visible_post_ids_json"])
                feed_ids = sorted(int(post["post_id"]) for post in feed.get("posts", []))
                feed_consistent += feed_ids == sorted(visible)
                if snapshot["status"] == "succeeded" and snapshot["action_trace_rowid"] is not None:
                    trace_bound += 1
                    for token in json.loads(snapshot["behavior_history_json"]):
                        trace_id = int(token.split(":", 1)[0].removeprefix("trace-"))
                        if trace_id >= int(snapshot["action_trace_rowid"]):
                            leakage_errors.append(str(snapshot["decision_id"]))
    scanned_files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "dataset_validation.json"
    ]
    secret_scan = not any(
        SECRET_PATTERN_BYTES.search(path.read_bytes()) for path in scanned_files
    )
    return {
        "action_label_eligible": sum(row.get("action_label_eligible", False) for row in succeeded),
        "decision_snapshot_count": database_snapshots,
        "failed_snapshot_count": len(rows) - len(succeeded),
        "feed_consistency_rate": feed_consistent / database_snapshots if database_snapshots else 0.0,
        "label_leakage_check_passed": not leakage_errors,
        "label_leakage_decision_ids": sorted(set(leakage_errors)),
        "rationale_fact_consistency_review_queue": len(succeeded),
        "rationale_training_eligible": sum(row.get("rationale_training_eligible", False) for row in succeeded),
        "secret_scan_passed": secret_scan,
        "succeeded_snapshot_count": len(succeeded),
        "teacher_candidate_count": len(succeeded),
        "trace_binding_rate": trace_bound / len(succeeded) if succeeded else 0.0,
        "unique_decision_id_count": len(set(ids)),
    }


def _audit_markdown(
    backend: BackendName,
    reliability: dict[str, Any],
    quality: dict[str, Any],
    dataset: dict[str, Any],
) -> str:
    return (
        "# Teacher evaluation audit\n\n"
        f"Backend: `{backend}`\n\n"
        "This benchmark uses a synthetic rule-based evaluation rubric, not a "
        "human-behavior gold standard. This output contains "
        f"{reliability['logical_decisions'] // 5} independent selected scenarios, "
        "five agents per scenario, and one decision per agent.\n\n"
        f"- Completed decisions: {reliability['completed_decisions']}/"
        f"{reliability['logical_decisions']}\n"
        f"- Aligned / acceptable / concerning / unscored: "
        f"{quality['aligned_count']} / {quality['acceptable_count']} / "
        f"{quality['concerning_count']} / {quality['unscored_count']}\n"
        f"- Native DecisionSnapshots: {dataset['decision_snapshot_count']}\n"
        f"- Teacher candidates: {dataset['teacher_candidate_count']}\n"
        f"- Secret scan passed: {str(dataset['secret_scan_passed']).lower()}\n\n"
        "Rationales remain in the manual-review queue and are not declared "
        "training-eligible by this local engineering validation.\n"
    )


def _database_integrity(path: Path, relative_path: str) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        snapshot_count, succeeded_count = connection.execute(
            "SELECT COUNT(*), SUM(status = 'succeeded') "
            "FROM diffusionguard_decision_snapshot"
        ).fetchone()
    return {
        "path": relative_path,
        "sha256": _sha256_file(path),
        "snapshot_count": int(snapshot_count),
        "succeeded_snapshot_count": int(succeeded_count or 0),
    }


def _complete_payload(
    *,
    batch_dir: Path,
    batch: dict[str, Any],
    backend: BackendName,
    config_sha256: str,
    rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    scenario_ids = [str(value) for value in batch["scenarios"]]
    databases = [
        _database_integrity(
            batch_dir / "scenarios" / f"{scenario_id}.db",
            f"scenarios/{scenario_id}.db",
        )
        for scenario_id in scenario_ids
    ]
    return {
        "actual_completed_decisions": sum(
            row.get("status") == "succeeded" for row in rows
        ),
        "backend": backend,
        "batch_id": str(batch["batch_id"]),
        "config_sha256": config_sha256,
        "expected_logical_decisions": len(scenario_ids) * 5,
        "integrity": {
            "databases": databases,
            "per_decision": {
                "rows": len(rows),
                "sha256": _sha256_file(batch_dir / "per_decision.jsonl"),
            },
            "per_scenario": {
                "rows": len(scenarios),
                "sha256": _sha256_file(batch_dir / "per_scenario.jsonl"),
            },
        },
        "max_physical_requests": MAX_PHYSICAL_REQUESTS_PER_BATCH,
        "response_cache_namespace": f"{batch['batch_id']}-isolated",
        "runtime": runtime,
        "scenario_ids": scenario_ids,
    }


def _resume_batch(
    *,
    batch_dir: Path,
    batch: dict[str, Any],
    backend: BackendName,
    config_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    complete_path = batch_dir / "complete.json"
    complete = _json_file(complete_path)
    if complete.get("config_sha256") != config_sha256:
        raise ResumeIntegrityError(
            f"resume refused for {batch['batch_id']}: suite config SHA-256 changed"
        )
    scenario_ids = [str(value) for value in batch["scenarios"]]
    expected = len(scenario_ids) * 5
    fixed_fields = {
        "backend": backend,
        "batch_id": str(batch["batch_id"]),
        "expected_logical_decisions": expected,
        "max_physical_requests": MAX_PHYSICAL_REQUESTS_PER_BATCH,
        "scenario_ids": scenario_ids,
    }
    for key, expected_value in fixed_fields.items():
        if complete.get(key) != expected_value:
            raise ValueError(f"complete metadata mismatch: {key}")

    decision_path = batch_dir / "per_decision.jsonl"
    scenario_path = batch_dir / "per_scenario.jsonl"
    integrity = complete.get("integrity")
    if not isinstance(integrity, dict):
        raise TypeError("complete integrity summary is missing")
    decision_integrity = integrity.get("per_decision", {})
    scenario_integrity = integrity.get("per_scenario", {})
    if (
        not decision_path.is_file()
        or _sha256_file(decision_path) != decision_integrity.get("sha256")
        or not scenario_path.is_file()
        or _sha256_file(scenario_path) != scenario_integrity.get("sha256")
    ):
        raise ValueError("batch JSONL integrity check failed")
    rows = _jsonl_file(decision_path)
    scenarios = _jsonl_file(scenario_path)
    if (
        len(rows) != expected
        or decision_integrity.get("rows") != expected
        or len({str(row.get("decision_id")) for row in rows}) != expected
        or {str(row.get("scenario_id")) for row in rows} != set(scenario_ids)
        or {str(row.get("backend")) for row in rows} != {backend}
        or len(scenarios) != len(scenario_ids)
        or scenario_integrity.get("rows") != len(scenario_ids)
        or {str(row.get("backend")) for row in scenarios} != {backend}
    ):
        raise ValueError("batch row-count or identity integrity check failed")
    completed = sum(row.get("status") == "succeeded" for row in rows)
    if complete.get("actual_completed_decisions") != completed:
        raise ValueError("batch completed-decision count disagrees with JSONL")

    database_entries = integrity.get("databases")
    if not isinstance(database_entries, list) or len(database_entries) != len(
        scenario_ids
    ):
        raise ValueError("database integrity summary is incomplete")
    by_path = {
        str(entry.get("path")): entry
        for entry in database_entries
        if isinstance(entry, dict)
    }
    database_succeeded = 0
    for scenario_id in scenario_ids:
        relative = f"scenarios/{scenario_id}.db"
        database = batch_dir / relative
        entry = by_path.get(relative)
        if entry is None or not database.is_file():
            raise ValueError("scenario database is missing from integrity summary")
        observed = _database_integrity(database, relative)
        if observed != entry or observed["snapshot_count"] != 5:
            raise ValueError("scenario database integrity check failed")
        database_succeeded += int(observed["succeeded_snapshot_count"])
    if database_succeeded != completed:
        raise ValueError("database success count disagrees with decision JSONL")
    return rows, scenarios, dict(complete.get("runtime", {}))


def _quarantine_batch(
    output: Path,
    batch_dir: Path,
    batch_id: str,
    now_factory: Callable[[], datetime],
) -> Path:
    timestamp = now_factory().astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = output / "quarantine" / f"{batch_id}-{timestamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"quarantine target already exists: {target}")
    shutil.move(str(batch_dir), str(target))
    return target


def _execution_status(
    reliability: dict[str, Any], expected_decisions: int
) -> tuple[str, int, list[str]]:
    completed = int(reliability["completed_decisions"])
    unrecovered = int(reliability["unrecovered_error_count"])
    reasons: list[str] = []
    if completed != expected_decisions:
        reasons.append("incomplete_decisions")
    if unrecovered:
        reasons.append("unrecovered_errors")
    if completed == expected_decisions and not reasons:
        status = "success"
    elif completed == 0:
        status = "failed"
    else:
        status = "degraded"
    return status, EXIT_CODES[status], reasons


async def run_suite(
    suite_path: Path,
    backend: BackendName,
    output: Path,
    *,
    selected_batch: str | None = None,
    resume: bool = False,
    llm_settings: LLMSettings | None = None,
    quarantine_now_factory: Callable[[], datetime] | None = None,
    model_factory: Callable[[LLMSettings], ManagedModelBackend | None] = (
        create_llm_model
    ),
) -> dict[str, Any]:
    """Run a complete offline suite or resume only at completed batch boundaries."""
    suite = load_suite(suite_path)
    selected_batches = _select_batches(suite, backend, selected_batch)
    available_batches = _available_batches(suite)
    config_sha256 = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    if backend == "groq":
        if llm_settings is None:
            raise RuntimeError("groq evaluation requires explicit LLM settings")
        if not llm_settings.enabled or llm_settings.provider != "groq":
            raise RuntimeError(
                "groq evaluation requires enabled, provider=groq LLM settings"
            )
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    now_factory = quarantine_now_factory or (lambda: datetime.now(UTC))
    scenarios_by_id = {str(s["scenario_id"]): s for s in suite["scenarios"]}
    profiles = list(suite["profiles"])
    all_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    manifest_batches: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    resumed_batches: dict[
        str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    ] = {}
    if resume:
        for batch in selected_batches:
            batch_id = str(batch["batch_id"])
            batch_dir = output / "batches" / batch_id
            if not batch_dir.exists():
                continue
            complete = batch_dir / "complete.json"
            if complete.is_file():
                try:
                    resumed_batches[batch_id] = _resume_batch(
                        batch_dir=batch_dir,
                        batch=batch,
                        backend=backend,
                        config_sha256=config_sha256,
                    )
                    continue
                except ResumeIntegrityError:
                    raise
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            _quarantine_batch(output, batch_dir, batch_id, now_factory)

    for batch in selected_batches:
        batch_id = str(batch["batch_id"])
        batch_dir = output / "batches" / batch_id
        if batch_id in resumed_batches:
            batch_rows, batch_scenarios, runtime_stats = resumed_batches[batch_id]
        else:
            if batch_dir.exists():
                raise RuntimeError(
                    f"batch directory unexpectedly exists after resume preflight: {batch_id}"
                )
            (batch_dir / "scenarios").mkdir(parents=True)
            batch_rows = []
            batch_scenarios = []
            model: ManagedModelBackend | None = None
            if backend == "groq":
                assert llm_settings is not None
                bounded = replace(
                    llm_settings,
                    temperature=0.0,
                    max_tokens=512,
                    max_retries=0,
                    max_concurrency=1,
                    max_calls_per_run=min(
                        llm_settings.max_calls_per_run,
                        MAX_PHYSICAL_REQUESTS_PER_BATCH,
                    ),
                    cache_enabled=True,
                    cache_path=batch_dir / "response_cache.sqlite3",
                )
                model = model_factory(bounded)
                if model is None:
                    raise RuntimeError("groq model creation was disabled")
            for scenario_id in batch["scenarios"]:
                scenario = scenarios_by_id[str(scenario_id)]
                db_path = batch_dir / "scenarios" / f"{scenario_id}.db"
                temporary_db = db_path.with_suffix(".db.pending")
                rows, summary = await _run_scenario(
                    scenario,
                    profiles,
                    backend,
                    temporary_db,
                    batch_id,
                    model=model,
                )
                temporary_db.replace(db_path)
                summary["batch_id"] = batch_id
                batch_rows.extend(rows)
                batch_scenarios.append(summary)
            batch_rows.sort(key=lambda row: row["decision_id"])
            batch_scenarios.sort(key=lambda row: row["scenario_id"])
            _write_jsonl(batch_dir / "per_decision.jsonl", batch_rows)
            _write_jsonl(batch_dir / "per_scenario.jsonl", batch_scenarios)
            runtime_stats = model.runtime.stats.to_dict() if model is not None else {}
            _write_json(
                batch_dir / "complete.json",
                _complete_payload(
                    batch_dir=batch_dir,
                    batch=batch,
                    backend=backend,
                    config_sha256=config_sha256,
                    rows=batch_rows,
                    scenarios=batch_scenarios,
                    runtime=runtime_stats,
                ),
            )
        all_rows.extend(batch_rows)
        scenario_summaries.extend(batch_scenarios)
        batch_runtime.append(runtime_stats)
        manifest_batches.append(
            {
                "batch_id": batch_id,
                "logical_decisions": len(batch_rows),
                "max_physical_requests": MAX_PHYSICAL_REQUESTS_PER_BATCH,
                "response_cache_path": f"batches/{batch_id}/response_cache.sqlite3",
                "scenarios": list(batch["scenarios"]),
            }
        )
    all_rows.sort(key=lambda row: row["decision_id"])
    scenario_summaries.sort(key=lambda row: row["scenario_id"])
    reliability = _reliability_summary(all_rows, backend, batch_runtime)
    quality = _quality_summary(all_rows)
    expected_decisions = sum(len(batch["scenarios"]) * 5 for batch in selected_batches)
    status, exit_code, status_reasons = _execution_status(
        reliability, expected_decisions
    )
    manifest = {
        "available_batches": available_batches,
        "backend": backend,
        "batches": manifest_batches,
        "config_sha256": config_sha256,
        "logical_decisions": len(all_rows),
        "max_physical_requests": (
            MAX_PHYSICAL_REQUESTS_PER_BATCH * len(selected_batches)
        ),
        "partial_suite": selected_batch is not None,
        "remote_api_calls": int(reliability["physical_remote_attempts"]),
        "scenario_count": len(scenario_summaries),
        "selected_batch": selected_batch,
        "suite_id": str(suite["suite_id"]),
        "suite_version": int(suite["version"]),
    }
    summary = {
        "completed_decisions": int(reliability["completed_decisions"]),
        "exit_code": exit_code,
        "expected_decisions": expected_decisions,
        "logical_decisions": len(all_rows),
        "selected_batch": selected_batch,
        "status": status,
        "status_reasons": status_reasons,
    }
    _write_jsonl(output / "per_decision.jsonl", all_rows)
    _write_jsonl(output / "per_scenario.jsonl", scenario_summaries)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "reliability_summary.json", reliability)
    _write_json(output / "quality_summary.json", quality)
    _write_json(output / "summary.json", summary)
    review_queue = [
        {
            "decision_id": row["decision_id"],
            "rationale": row["rationale"],
            "review_reason": "factual_consistency_not_human_reviewed",
            "scenario_id": row["scenario_id"],
        }
        for row in all_rows
        if row["status"] == "succeeded"
    ]
    _write_jsonl(output / "rationale_review_queue.jsonl", review_queue)
    dataset = _dataset_validation(all_rows, scenario_summaries, output)
    _write_json(output / "dataset_validation.json", dataset)
    _atomic_text(
        output / "AUDIT_REPORT.md",
        _audit_markdown(backend, reliability, quality, dataset),
    )
    return {
        "dataset_validation": dataset,
        "manifest": manifest,
        "quality": quality,
        "reliability": reliability,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("fake", "random_legal", "deterministic_risk_rule", "groq"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch",
        help="run exactly one configured batch (required for backend=groq)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print a non-sensitive plan without creating output or loading env",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="loaded only for backend=groq; local backends never read it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        plan = build_plan(
            args.suite,
            args.backend,
            args.batch,
            args.output,
            resume=args.resume,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"teacher evaluation configuration failed: {exc}") from None
    if args.plan:
        print(_canonical(plan, pretty=True), end="")
        return
    if args.backend == "groq":
        if not args.env_file.is_file():
            raise SystemExit("groq backend requires an explicit local env file")
        load_dotenv(args.env_file, override=False)
    settings = LLMSettings.from_env() if args.backend == "groq" else None
    try:
        result = asyncio.run(
            run_suite(
                args.suite,
                args.backend,
                args.output,
                selected_batch=args.batch,
                resume=args.resume,
                llm_settings=settings,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"teacher evaluation failed: {exc}") from None
    print(_canonical(result, pretty=True), end="")
    raise SystemExit(int(result["summary"]["exit_code"]))


if __name__ == "__main__":
    main()
