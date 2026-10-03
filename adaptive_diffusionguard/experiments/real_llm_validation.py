"""Bounded OASIS validation with legal masks and strict structured outputs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from oasis import generate_twitter_agent_graph, make
from oasis.social_platform.typing import ActionType, RecsysType
from pydantic import ValidationError

from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy
from adaptive_diffusionguard.llm.action_gateway import (
    ActionDecisionError,
    GatewayAudit,
    attach_action_gateway,
)
from adaptive_diffusionguard.llm.actions import ActionOutput
from adaptive_diffusionguard.llm.model_factory import LLMSettings, create_llm_model
from adaptive_diffusionguard.llm.runtime import (
    CallLimitExceeded,
    ManagedModelBackend,
    is_json_validate_failed,
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
    StructuredAttemptResult,
    request_with_structured_correction,
)
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform
from adaptive_diffusionguard.storage.decision_snapshots import DecisionSnapshotStore
from training.build_dataset import examples_from_oasis, write_jsonl
from training.teacher import redact_sensitive

AGENT_COUNT = 5
TIMESTEPS = 3
MAX_CALLS = 30
TEACHER_MAX_TOKENS = 512
TEACHER_TEMPERATURE = 0.0
SEED = 20260927
EXIT_CODES = {"success": 0, "failed": 1, "degraded": 2}
ACTION_MAP = {
    "repost": "repost",
    "quote_post": "quote",
    "report_post": "report",
    "do_nothing": "ignore",
}


def _write_profiles(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["username", "description", "user_char"]
        )
        writer.writeheader()
        for agent_id in range(AGENT_COUNT):
            writer.writerow(
                {
                    "username": f"synthetic_agent_{agent_id}",
                    "description": "Synthetic research agent; no personal data.",
                    "user_char": (
                        "Evaluate claims cautiously and choose one legal action "
                        "from the current feed."
                    ),
                }
            )


def _sanitize_examples(examples, rationales: dict[str, str]):
    for example in examples:
        rationale = rationales.get(example.sample_id, example.label.reason)
        yield replace(
            example,
            user_profile=redact_sensitive(example.user_profile),
            feed_post=redact_sensitive(example.feed_post),
            neighbor_interactions=[
                redact_sensitive(v) for v in example.neighbor_interactions
            ],
            behavior_history=[redact_sensitive(v) for v in example.behavior_history],
            platform_notice=redact_sensitive(example.platform_notice),
            label=ActionOutput(
                action=example.label.action,
                confidence=example.label.confidence,
                reason=redact_sensitive(rationale),
            ),
            label_source="teacher_synthetic",
        )


def _outputs_are_secret_free(output: Path, settings: LLMSettings) -> bool:
    """Scan outputs without emitting credentials or credential fragments."""
    secrets = [
        value.encode()
        for value in (settings.groq_api_key, settings.openai_api_key)
        if value
    ]
    if not secrets:
        return True
    clean = True
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if any(secret in payload for secret in secrets):
            path.unlink()
            clean = False
    return clean


def _safe_error_message(exc: BaseException, secrets: list[str]) -> str:
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)\b(?:gsk_|sk-|hf_)[A-Za-z0-9_-]+", "[REDACTED]", message)


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _determine_status(
    *,
    logical_decisions: int,
    completed_decisions: int,
    impression_count: int,
    governance_executed: bool,
    secret_scan_passed: bool,
    runtime: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(runtime.get("successful_provider_responses", 0)) == 0:
        reasons.append("no_successful_provider_responses")
    if impression_count == 0:
        reasons.append("no_impression_records")
    if not governance_executed:
        reasons.append("governance_not_executed")
    if not secret_scan_passed:
        reasons.append("output_secret_scan_failed")
    if reasons:
        return "failed", reasons

    if completed_decisions != logical_decisions:
        reasons.append("incomplete_agent_decisions")
    statuses = runtime.get("unrecovered_http_statuses", {})
    for status, count in statuses.items():
        code = int(status)
        if count and (code in {401, 403, 404, 429} or code >= 500):
            reasons.append(f"unrecovered_http_{code}")
    if runtime.get("call_limit_exceeded_count", 0):
        reasons.append("physical_call_limit_exceeded")
    if (
        runtime.get("physical_remote_attempts", 0)
        >= runtime.get("max_calls_per_run", float("inf"))
        and completed_decisions < logical_decisions
    ):
        reasons.append("physical_call_limit_reached_with_incomplete_decisions")
    if runtime.get("empty_response_count", 0):
        reasons.append("empty_provider_response")
    if runtime.get("unrecovered_error_count", 0) and not statuses:
        reasons.append("unrecovered_provider_error")
    return ("degraded", sorted(set(reasons))) if reasons else ("success", [])


def _profile_for(agent: Any) -> str:
    description = getattr(agent.user_info, "description", "")
    return (
        f"synthetic agent {agent.social_agent_id}; {description or 'research profile'}"
    )


def _latest_trace_rowid(platform: AdaptiveDiffusionPlatform) -> int:
    row = platform.db.execute("SELECT COALESCE(MAX(rowid), 0) FROM trace").fetchone()
    return int(row[0])


def _behavior_history_before(
    platform: AdaptiveDiffusionPlatform, user_id: int
) -> list[str]:
    rows = platform.db.execute(
        "SELECT rowid, action FROM trace WHERE user_id = ? "
        "ORDER BY rowid DESC LIMIT 10",
        (int(user_id),),
    ).fetchall()
    return [f"trace-{int(rowid)}:{action}" for rowid, action in reversed(rows)]


def _decision_failure_category(exc: BaseException) -> str:
    """Return only a stable, non-sensitive failure category."""
    if is_json_validate_failed(exc):
        return "json_validate_failed"
    if isinstance(exc, CallLimitExceeded):
        return "call_limit_exceeded"
    if isinstance(exc, ActionDecisionError):
        return "action_decision_error"
    if isinstance(exc, ValidationError):
        return "schema_validation_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return "value_error"
    return "provider_or_runtime_error"


def _dispatched_trace_row(
    platform: AdaptiveDiffusionPlatform,
    *,
    after_rowid: int,
    user_id: int,
    choice_id: str,
) -> tuple[int, str] | None:
    action_name = choice_id.partition(":")[0]
    expected_oasis_action = {
        "repost": "repost",
        "quote": "quote_post",
        "report": "report_post",
        "ignore": "do_nothing",
    }[action_name]
    row = platform.db.execute(
        "SELECT rowid, action FROM trace WHERE rowid > ? AND user_id = ? "
        "AND action = ? ORDER BY rowid LIMIT 1",
        (after_rowid, user_id, expected_oasis_action),
    ).fetchone()
    return (int(row[0]), str(row[1])) if row is not None else None


async def run_validation(
    settings: LLMSettings,
    output: Path,
    *,
    model_override: ManagedModelBackend | None = None,
) -> dict[str, Any]:
    """Run 5 x 3 decisions; a supplied model enables a no-network integration run."""
    bounded = replace(
        settings,
        temperature=TEACHER_TEMPERATURE,
        max_tokens=TEACHER_MAX_TOKENS,
        max_calls_per_run=min(settings.max_calls_per_run, MAX_CALLS),
        max_concurrency=1,
        max_retries=0,
        request_interval_seconds=max(settings.request_interval_seconds, 8.0),
        cache_enabled=True,
    )
    model = model_override or create_llm_model(bounded)
    if model is None:
        raise RuntimeError("real validation requires DIFFUSIONGUARD_ENABLE_LLM=true")

    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "synthetic_profiles.csv"
    db_path = output / "real_llm_validation.db"
    teacher_path = output / "groq_teacher_trajectories.jsonl"
    _write_profiles(profile_path)
    if db_path.exists():
        db_path.unlink()

    # OASIS interprets an empty list as all actions. The placeholder is removed
    # immediately after reset; no FunctionTool is used for model decisions.
    graph = await generate_twitter_agent_graph(
        str(profile_path), model=model, available_actions=[ActionType.DO_NOTHING]
    )
    communities = {
        agent_id: f"community-{agent_id % 2}" for agent_id in range(AGENT_COUNT)
    }
    platform = AdaptiveDiffusionPlatform(
        str(db_path),
        user_communities=communities,
        post_risk_scores={},
        policy=StaticCOSREFPolicy(omega_intra=0.4, omega_inter=0.0, seed=SEED),
        random_seed=SEED,
        run_id=f"real-llm-{SEED}",
        recsys_type=RecsysType.RANDOM,
        refresh_rec_post_count=2,
        following_post_count=2,
        max_rec_post_len=2,
        intervention_budget=1.6,
    )
    environment = make(
        agent_graph=graph,
        platform=platform,
        database_path=str(db_path),
        semaphore=1,
    )
    await environment.reset()
    agents = [agent for _, agent in environment.agent_graph.get_agents()]
    gateways = [attach_action_gateway(agent) for agent in agents]
    mask_builder = ActionMaskBuilder(platform.db)
    decision_store = DecisionSnapshotStore(platform.db)
    structured_response_count = 0
    structured_schema_valid_count = 0
    legal_choice_count = 0
    action_mask_empty_count = 0
    decision_errors: list[dict[str, str]] = []
    completed_timesteps = 0
    decision_sequence = 0

    try:
        normal = await platform.create_post(0, "Verified synthetic weather update.")
        risky = await platform.create_post(1, "Unverified synthetic health claim.")
        platform.post_risk_scores[int(normal["post_id"])] = 0.0
        platform.post_risk_scores[int(risky["post_id"])] = 1.0

        for timestep in range(TIMESTEPS):
            platform.sandbox_clock.time_step = timestep
            await platform.update_rec_table()
            for agent, gateway in zip(agents, gateways, strict=True):
                decision_sequence += 1
                user_id = int(agent.social_agent_id)
                messages: list[dict[str, Any]] | None = None
                response_format: dict[str, Any] | None = None
                attempt_result: StructuredAttemptResult | None = None
                completion_recorded = False
                persisted_decision_id: str | None = None
                snapshot_terminal = False
                try:
                    feed = await agent.env.action.refresh()
                    gateway.update_visible_feed(feed)
                    mask = mask_builder.build(user_id, gateway.visible_post_ids)
                    if not mask.choices:
                        action_mask_empty_count += 1
                        raise ActionDecisionError("current legal action mask is empty")
                    response_format = build_response_format(mask)
                    messages = build_decision_messages(
                        profile=_profile_for(agent), feed=feed, mask=mask
                    )
                    snapshot = DecisionSnapshot.capture(
                        user_id=user_id,
                        timestep=timestep,
                        feed=feed,
                        visible_post_ids=gateway.visible_post_ids,
                        mask=mask,
                        state_identifier=mask_builder.state_identifier(
                            user_id, gateway.visible_post_ids
                        ),
                        messages=messages,
                        response_format=response_format,
                    )
                    pending = decision_store.create_pending(
                        snapshot,
                        run_id=platform.run_id,
                        agent_id=int(agent.social_agent_id),
                        decision_sequence=decision_sequence,
                        user_profile=_profile_for(agent),
                        community=communities[user_id],
                        behavior_history=_behavior_history_before(platform, user_id),
                        neighbor_interactions=[],
                        platform_notice="",
                    )
                    persisted_decision_id = pending.decision_id

                    attempt_result = await request_with_structured_correction(
                        model, snapshot
                    )
                    response = attempt_result.response
                    messages = attempt_result.request.messages
                    response_format = attempt_result.request.response_format
                    structured_response_count += 1
                    decision = parse_structured_response(response)
                    structured_schema_valid_count += 1
                    if decision.choice_id not in snapshot.legal_choice_ids:
                        gateway.audit.invalid_choice_count += 1
                        raise ActionDecisionError(
                            "choice_id is not in the original decision snapshot"
                        )

                    # The model is not called again if the persisted state
                    # changed. The original visible IDs remain authoritative.
                    try:
                        current_mask = revalidate_snapshot_choice(
                            snapshot, decision.choice_id, mask_builder
                        )
                    except ValueError as exc:
                        gateway.audit.invalid_choice_count += 1
                        raise ActionDecisionError(str(exc)) from None
                    legal_choice_count += 1
                    trace_rowid_before = _latest_trace_rowid(platform)
                    await gateway.dispatch_choice(
                        decision.choice_id, decision.rationale, current_mask.choices
                    )
                    trace_row = _dispatched_trace_row(
                        platform,
                        after_rowid=trace_rowid_before,
                        user_id=user_id,
                        choice_id=decision.choice_id,
                    )
                    if trace_row is None:
                        action_name = decision.choice_id.partition(":")[0]
                        gateway.audit.dispatcher_success_count -= 1
                        gateway.audit.action_counts[action_name] -= 1
                        gateway.audit.dispatcher_failure_count += 1
                        raise ActionDecisionError(
                            "OASIS dispatch returned success without a trace record"
                        )
                    decision_store.mark_succeeded(
                        persisted_decision_id,
                        selected_choice_id=decision.choice_id,
                        rationale=decision.rationale,
                        action_trace_rowid=trace_row[0],
                    )
                    snapshot_terminal = True
                    model.runtime.record_structured_completion(
                        retried=attempt_result.retried
                    )
                    completion_recorded = True
                    # Only a schema-valid, legal, successfully dispatched action
                    # can become a reusable cached teacher response.
                    model.cache_structured_response(
                        response,
                        messages=messages,
                        response_format=response_format,
                    )
                except (ActionDecisionError, ValidationError, ValueError) as exc:
                    if persisted_decision_id is not None and not snapshot_terminal:
                        decision_store.mark_failed(
                            persisted_decision_id, _decision_failure_category(exc)
                        )
                        snapshot_terminal = True
                    if (
                        attempt_result is not None
                        and attempt_result.retried
                        and not completion_recorded
                    ):
                        model.runtime.record_structured_retry_failure()
                    if messages is not None and response_format is not None:
                        model.discard_structured_response(
                            messages=messages, response_format=response_format
                        )
                    decision_errors.append(
                        {"type": type(exc).__name__, "message": str(exc)}
                    )
                except Exception as exc:  # noqa: BLE001 - provider/runtime boundary
                    if persisted_decision_id is not None and not snapshot_terminal:
                        decision_store.mark_failed(
                            persisted_decision_id, _decision_failure_category(exc)
                        )
                        snapshot_terminal = True
                    decision_errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": _safe_error_message(
                                exc,
                                [bounded.groq_api_key, bounded.openai_api_key],
                            ),
                        }
                    )
            completed_timesteps += 1

        impression_total, suppressed = platform.db.execute(
            "SELECT COUNT(*), SUM(shown = 0) FROM diffusionguard_impression"
        ).fetchone()
        action_rows = platform.db.execute(
            "SELECT action FROM trace WHERE action IN (?, ?, ?, ?)",
            tuple(ACTION_MAP),
        ).fetchall()
        trace_action_counts = Counter(ACTION_MAP[row[0]] for row in action_rows)
        decision_snapshot_status_counts = decision_store.status_counts()
    finally:
        await environment.close()

    profiles = {str(i): f"synthetic research agent {i}" for i in range(AGENT_COUNT)}
    community_map = {str(key): value for key, value in communities.items()}
    examples = list(
        examples_from_oasis(
            db_path, profiles, community_map, label_source="teacher_synthetic"
        )
    )
    count = write_jsonl(_sanitize_examples(examples, {}), teacher_path)
    secret_scan_passed = _outputs_are_secret_free(output, bounded)
    runtime = model.runtime.stats.to_dict()
    runtime["max_calls_per_run"] = bounded.max_calls_per_run
    gateway_metrics = GatewayAudit.merge(
        gateway.audit for gateway in gateways
    ).to_dict()
    logical_decisions = AGENT_COUNT * TIMESTEPS
    completed_decisions = int(gateway_metrics["dispatcher_success_count"])
    governance_executed = bool(impression_total)
    status, status_reasons = _determine_status(
        logical_decisions=logical_decisions,
        completed_decisions=completed_decisions,
        impression_count=int(impression_total or 0),
        governance_executed=governance_executed,
        secret_scan_passed=secret_scan_passed,
        runtime=runtime,
    )
    physical_attempts = int(runtime["physical_remote_attempts"])
    successful_responses = int(runtime["successful_provider_responses"])
    summary = {
        "status": status,
        "exit_code": EXIT_CODES[status],
        "status_reasons": status_reasons,
        "validation_mode": "strict_structured_action_mask",
        "provider": bounded.provider,
        "model": bounded.llm_model,
        "agents": AGENT_COUNT,
        "timesteps": TIMESTEPS,
        "max_calls": bounded.max_calls_per_run,
        "logical_decisions": logical_decisions,
        "structured_response_count": structured_response_count,
        "structured_schema_valid_count": structured_schema_valid_count,
        "legal_choice_count": legal_choice_count,
        "dispatcher_success_count": completed_decisions,
        "invalid_choice_count": int(gateway_metrics["invalid_choice_count"]),
        "invalid_post_reference_count": int(
            gateway_metrics["invalid_post_reference_count"]
        ),
        "action_mask_empty_count": action_mask_empty_count,
        "dispatcher_failure_count": int(gateway_metrics["dispatcher_failure_count"]),
        "completed_decisions": completed_decisions,
        "action_counts": gateway_metrics["action_counts"],
        "provider_successful_responses": successful_responses,
        "successful_provider_responses": successful_responses,
        "physical_remote_attempts": physical_attempts,
        "cache_hits": int(runtime["cache_hits"]),
        "retry_count": int(runtime["retry_count"]),
        "first_attempt_count": int(runtime["first_attempt_count"]),
        "first_attempt_success_count": int(runtime["first_attempt_success_count"]),
        "first_attempt_success_rate": float(runtime["first_attempt_success_rate"]),
        "json_validate_failed_count": int(runtime["json_validate_failed_count"]),
        "structured_retry_attempt_count": int(
            runtime["structured_retry_attempt_count"]
        ),
        "structured_retry_success_count": int(
            runtime["structured_retry_success_count"]
        ),
        "structured_retry_failure_count": int(
            runtime["structured_retry_failure_count"]
        ),
        "post_retry_completed_decisions": int(
            runtime["post_retry_completed_decisions"]
        ),
        "post_retry_success_rate": float(runtime["post_retry_success_rate"]),
        "rate_limit_count": int(runtime["rate_limit_count"]),
        "unrecovered_error_count": int(runtime["unrecovered_error_count"]),
        "unrecovered_http_statuses": runtime["unrecovered_http_statuses"],
        "prompt_tokens": int(runtime["prompt_tokens"]),
        "completion_tokens": int(runtime["completion_tokens"]),
        "total_tokens": int(runtime["total_tokens"]),
        "teacher_examples": count,
        "decision_snapshot_status_counts": decision_snapshot_status_counts,
        "secret_scan_passed": secret_scan_passed,
        # Compatibility fields are intentionally zero in structured mode.
        "tool_call_count": int(runtime["tool_call_count"]),
        "raw_tool_call_count": 0,
        "provider_tool_use_failed_count": int(
            runtime["provider_tool_use_failed_count"]
        ),
        "no_tool_call_response_count": int(runtime["no_tool_call_response_count"]),
        "empty_response_count": int(runtime["empty_response_count"]),
        "completed_timesteps": completed_timesteps,
        "runtime": runtime,
        "impression_count": int(impression_total or 0),
        "suppressed_impressions": int(suppressed or 0),
        "governance_executed": governance_executed,
        "trace_action_counts": dict(trace_action_counts),
        "teacher_jsonl": str(teacher_path),
        "provider_success_rate": (
            successful_responses / physical_attempts if physical_attempts else 0.0
        ),
        "json_valid_rate": structured_schema_valid_count / logical_decisions,
        "action_valid_rate": completed_decisions / logical_decisions,
        "teacher_temperature": bounded.temperature,
        "teacher_max_tokens": bounded.max_tokens,
        "decision_errors": decision_errors,
    }
    _write_summary(output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("runs/real-llm"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if not args.env_file.is_file():
            raise RuntimeError(
                "real validation skipped: no local .env file was detected; no API call made"
            )
        load_dotenv(args.env_file, override=False)
        if (
            not os.environ.get("GROQ_API_KEY")
            and os.environ.get("LLM_PROVIDER", "groq") == "groq"
        ):
            raise RuntimeError(
                "real validation skipped: GROQ_API_KEY is absent; no API call made"
            )
        settings = LLMSettings.from_env()
        summary = asyncio.run(run_validation(settings, args.output))
    except Exception as exc:  # noqa: BLE001 - CLI boundary must emit failed summary
        secrets = [
            os.environ.get("GROQ_API_KEY", ""),
            os.environ.get("OPENAI_API_KEY", ""),
        ]
        summary = {
            "status": "failed",
            "exit_code": EXIT_CODES["failed"],
            "status_reasons": ["configuration_or_runtime_failure"],
            "error": {
                "type": type(exc).__name__,
                "message": _safe_error_message(exc, secrets),
            },
        }
        _write_summary(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(int(summary["exit_code"]))


if __name__ == "__main__":
    main()
