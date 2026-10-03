"""Leakage-resistant offline builders for native and legacy teacher data."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_diffusionguard.storage.decision_snapshots import canonical_json
from training.schemas import ActionOutput, BehaviorExample, LabelSource

SNAPSHOT_TABLE = "diffusionguard_decision_snapshot"
ACTION_MAP = {
    "repost": "repost",
    "quote_post": "quote",
    "report_post": "report",
    "do_nothing": "ignore",
}
OASIS_ACTION = {value: key for key, value in ACTION_MAP.items()}
SECRET_PATTERN = re.compile(r"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")
ACTION_RESULT_KEYS = frozenset({"new_post_id", "quoted_id", "reposted_id", "report_id"})


class DatasetIntegrityError(RuntimeError):
    """Raised when a dataset would be ambiguous, incomplete, or label-leaking."""


@dataclass(frozen=True, slots=True)
class DatasetBuild:
    examples: tuple[BehaviorExample, ...]
    report: dict[str, Any]


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def has_decision_snapshots(db_path: Path) -> bool:
    connection = _open_read_only(db_path)
    try:
        return _table_exists(connection, SNAPSHOT_TABLE)
    finally:
        connection.close()


def _json_object(raw: str, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIntegrityError(f"{field} must be a JSON object")
    return value


def _json_list(raw: str, *, field: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise DatasetIntegrityError(f"{field} must be a JSON array")
    return value


def _feed_post_ids(feed: dict[str, Any]) -> list[int]:
    posts = feed.get("posts")
    if not isinstance(posts, list):
        raise DatasetIntegrityError("feed_json.posts must be an array")
    ids: list[int] = []
    for post in posts:
        if not isinstance(post, dict) or post.get("post_id") is None:
            raise DatasetIntegrityError("feed_json contains a post without post_id")
        ids.append(int(post["post_id"]))
    if len(ids) != len(set(ids)):
        raise DatasetIntegrityError("feed_json contains duplicate post_id values")
    return ids


def _choice_action(choice_id: str) -> str:
    action = choice_id.partition(":")[0]
    if action not in OASIS_ACTION:
        raise DatasetIntegrityError(f"unsupported selected choice action: {action}")
    return action


def _choice_target(choice_id: str) -> int | None:
    action, separator, raw_target = choice_id.partition(":")
    if action == "ignore":
        if separator:
            raise DatasetIntegrityError("ignore choice must not have a target")
        return None
    if not separator:
        raise DatasetIntegrityError("targeted choice is missing post_id")
    try:
        target = int(raw_target)
    except ValueError as exc:
        raise DatasetIntegrityError("choice target is not an integer") from exc
    if target <= 0:
        raise DatasetIntegrityError("choice target must be positive")
    return target


def _trace_choice(oasis_action: str, info: dict[str, Any]) -> str:
    action = ACTION_MAP[oasis_action]
    if action == "ignore":
        return "ignore"
    keys = {
        "repost": ("reposted_id", "post_id"),
        "quote": ("quoted_id", "post_id", "original_post_id"),
        "report": ("post_id",),
    }[action]
    target: Any = None
    for key in keys:
        if info.get(key) is not None:
            target = info[key]
            break
    if target is None:
        raise DatasetIntegrityError(f"{oasis_action} trace has no target post")
    return f"{action}:{int(target)}"


def _validate_history(history: list[str], action_trace_rowid: int) -> None:
    for token in history:
        match = re.fullmatch(r"trace-(\d+):([a-z_]+)", token)
        if match is None:
            raise DatasetIntegrityError(f"history token has invalid format: {token}")
        if int(match.group(1)) >= action_trace_rowid:
            raise DatasetIntegrityError(
                "behavior_history contains the current or a future trace"
            )


def _validate_snapshot_row(
    connection: sqlite3.Connection, row: sqlite3.Row
) -> BehaviorExample:
    decision_id = str(row["decision_id"])
    if row["status"] != "succeeded":
        raise DatasetIntegrityError(
            f"decision {decision_id} is not succeeded: {row['status']}"
        )
    required = ("selected_choice_id", "action_trace_rowid", "rationale")
    missing = [name for name in required if row[name] is None]
    if missing:
        raise DatasetIntegrityError(
            f"decision {decision_id} lacks succeeded fields: {', '.join(missing)}"
        )
    feed = _json_object(str(row["feed_json"]), field="feed_json")
    feed_ids = _feed_post_ids(feed)
    visible_ids = [
        int(value)
        for value in _json_list(
            str(row["visible_post_ids_json"]), field="visible_post_ids_json"
        )
    ]
    if sorted(feed_ids) != sorted(visible_ids):
        raise DatasetIntegrityError(
            f"decision {decision_id} feed and visible_post_ids disagree"
        )
    legal_choices = [
        str(value)
        for value in _json_list(
            str(row["legal_choice_ids_json"]), field="legal_choice_ids_json"
        )
    ]
    selected = str(row["selected_choice_id"])
    if selected not in legal_choices:
        raise DatasetIntegrityError(
            f"decision {decision_id} selected choice is not legal"
        )
    target = _choice_target(selected)
    if target is not None and target not in visible_ids:
        raise DatasetIntegrityError(
            f"decision {decision_id} selected target was not visible"
        )
    trace_rowid = int(row["action_trace_rowid"])
    trace = connection.execute(
        "SELECT user_id, action, info FROM trace WHERE rowid = ?", (trace_rowid,)
    ).fetchone()
    if trace is None:
        raise DatasetIntegrityError(
            f"decision {decision_id} action trace does not exist"
        )
    action = _choice_action(selected)
    if int(trace[0]) != int(row["user_id"]) or trace[1] != OASIS_ACTION[action]:
        raise DatasetIntegrityError(
            f"decision {decision_id} action trace binding is inconsistent"
        )
    trace_choice = _trace_choice(
        str(trace[1]), _json_object(trace[2], field="trace.info")
    )
    if trace_choice != selected:
        raise DatasetIntegrityError(
            f"decision {decision_id} selected choice disagrees with action trace"
        )
    history = [
        str(value)
        for value in _json_list(
            str(row["behavior_history_json"]), field="behavior_history_json"
        )
    ]
    _validate_history(history, trace_rowid)
    neighbors = [
        str(value)
        for value in _json_list(
            str(row["neighbor_interactions_json"]),
            field="neighbor_interactions_json",
        )
    ]
    rationale = str(row["rationale"])
    if SECRET_PATTERN.search(rationale):
        raise DatasetIntegrityError("rationale contains a credential-like pattern")
    empty_reason = None
    if not feed_ids:
        empty_reason = str(feed.get("message") or feed.get("error") or "") or None
        if empty_reason is None:
            raise DatasetIntegrityError(
                f"decision {decision_id} has an empty feed without a reason"
            )
    return BehaviorExample(
        sample_id=decision_id,
        user_profile=str(row["user_profile"]),
        community=str(row["community"]),
        feed_post=canonical_json(feed),
        neighbor_interactions=neighbors,
        behavior_history=history,
        platform_notice=str(row["platform_notice"]),
        label=ActionOutput(action=action, confidence=1.0, reason=rationale),
        label_source="teacher_synthetic",
        action_trace_rowid=trace_rowid,
        decision_sequence=int(row["decision_sequence"]),
        decision_timestep=int(row["timestep"]),
        visible_post_ids=visible_ids,
        legal_choice_ids=legal_choices,
        selected_choice_id=selected,
        provenance="decision_snapshot",
        action_label_eligible=True,
        rationale_training_eligible=True,
        rationale_quality_status="unreviewed",
        feed_empty_reason=empty_reason,
    )


def build_from_decision_snapshots(
    db_path: Path, *, allow_failed: bool = False
) -> DatasetBuild:
    """Build only from succeeded native snapshots, rejecting inconsistent rows."""
    connection = _open_read_only(db_path)
    try:
        if not _table_exists(connection, SNAPSHOT_TABLE):
            raise DatasetIntegrityError(
                "database has no diffusionguard_decision_snapshot table"
            )
        rows = connection.execute(
            f"SELECT * FROM {SNAPSHOT_TABLE} ORDER BY decision_sequence, decision_id"
        ).fetchall()
        status_counts = Counter(str(row["status"]) for row in rows)
        if status_counts["pending"]:
            raise DatasetIntegrityError("pending decision snapshots cannot be exported")
        if status_counts["failed"] and not allow_failed:
            raise DatasetIntegrityError(
                "failed decision snapshots require an explicitly degraded export"
            )
        succeeded = [row for row in rows if row["status"] == "succeeded"]
        examples = tuple(_validate_snapshot_row(connection, row) for row in succeeded)
    finally:
        connection.close()
    ids = [example.sample_id for example in examples]
    trace_ids = [example.action_trace_rowid for example in examples]
    if len(ids) != len(set(ids)) or len(trace_ids) != len(set(trace_ids)):
        raise DatasetIntegrityError("snapshot export contains duplicate bindings")
    report = {
        "action_label_eligible": sum(e.action_label_eligible for e in examples),
        "decision_snapshot_status_counts": {
            key: status_counts.get(key, 0) for key in ("failed", "pending", "succeeded")
        },
        "excluded_failed_decisions": status_counts["failed"],
        "output_examples": len(examples),
        "provenance": "decision_snapshot",
        "rationale_training_eligible": sum(
            e.rationale_training_eligible for e in examples
        ),
        "status": "degraded" if status_counts["failed"] else "success",
    }
    return DatasetBuild(examples=examples, report=report)


def _root_post_id(posts: dict[int, dict[str, Any]], post_id: int) -> int:
    current = post_id
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        post = posts.get(current)
        if post is None:
            raise DatasetIntegrityError(f"post {current} is missing")
        parent = post["original_post_id"]
        if parent is None:
            return current
        current = int(parent)
    raise DatasetIntegrityError(f"post root cycle starts at {post_id}")


def _legacy_legal_choices(
    *,
    user_id: int,
    visible_ids: list[int],
    posts: dict[int, dict[str, Any]],
    created_derivatives: set[int],
    reports: set[tuple[int, int]],
) -> list[str]:
    choices = ["ignore"]
    for post_id in sorted(visible_ids):
        post = posts.get(post_id)
        if post is None:
            raise DatasetIntegrityError(f"visible post {post_id} is missing")
        repost_targets = [post_id]
        if post["original_post_id"] is not None and post["quote_content"] is None:
            repost_targets.append(_root_post_id(posts, post_id))
        already_derived = any(
            derived_id in posts
            and int(posts[derived_id]["user_id"]) == user_id
            and posts[derived_id]["original_post_id"] in repost_targets
            for derived_id in created_derivatives
        )
        if not already_derived:
            choices.append(f"repost:{post_id}")
        choices.append(f"quote:{post_id}")
        if (user_id, post_id) not in reports:
            choices.append(f"report:{post_id}")
    return choices


def _load_teacher_source(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    payload = path.read_text(encoding="utf-8")
    if SECRET_PATTERN.search(payload):
        raise DatasetIntegrityError("teacher source contains a credential-like pattern")
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(
                f"teacher source line {line_number} is invalid JSON"
            ) from exc
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in rows:
            raise DatasetIntegrityError(
                f"teacher source line {line_number} has a missing or duplicate sample_id"
            )
        rows[sample_id] = row
    return rows


def _load_rationale_issues(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = path.read_text(encoding="utf-8")
    if SECRET_PATTERN.search(raw):
        raise DatasetIntegrityError("quality audit contains a credential-like pattern")
    payload = json.loads(raw)
    groups = payload.get("quality", {}).get("rationale_issue_sample_ids", {})
    issues: dict[str, str] = {}
    priority = (
        ("factual_author_mismatch", "factual_relation_error"),
        ("action_contradiction", "action_contradiction"),
        ("wrong_post_reference", "wrong_post_reference"),
        ("empty", "empty_rationale"),
        ("too_short", "too_short"),
    )
    for key, status in priority:
        for sample_id in groups.get(key, []):
            issues.setdefault(str(sample_id), status)
    return issues


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_legacy_dataset(
    db_path: Path,
    teacher_source_path: Path,
    *,
    quality_audit_path: Path | None = None,
    label_source: LabelSource = "teacher_synthetic",
) -> DatasetBuild:
    """Explicitly reconstruct pre-decision contexts from unambiguous legacy traces."""
    teacher_source = _load_teacher_source(teacher_source_path)
    rationale_issues = _load_rationale_issues(quality_audit_path)
    connection = _open_read_only(db_path)
    try:
        if _table_exists(connection, SNAPSHOT_TABLE):
            raise DatasetIntegrityError(
                "legacy reconstruction is forbidden when native snapshots exist"
            )
        traces = [
            dict(row)
            for row in connection.execute(
                "SELECT rowid, user_id, created_at, action, info "
                "FROM trace ORDER BY rowid"
            )
        ]
        posts = {
            int(row["post_id"]): dict(row)
            for row in connection.execute(
                "SELECT post_id, user_id, original_post_id, quote_content, content "
                "FROM post ORDER BY post_id"
            )
        }
        run_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT run_id FROM diffusionguard_impression ORDER BY run_id"
            )
        ]
        community_rows = connection.execute(
            "SELECT user_id, user_community FROM diffusionguard_impression "
            "GROUP BY user_id, user_community ORDER BY user_id"
        ).fetchall()
    finally:
        connection.close()
    if len(run_ids) != 1:
        raise DatasetIntegrityError("legacy database must contain exactly one run_id")
    community_values: dict[int, set[str]] = defaultdict(set)
    for user_id, community in community_rows:
        community_values[int(user_id)].add(str(community))
    conflicting = [
        user for user, values in community_values.items() if len(values) != 1
    ]
    if conflicting:
        raise DatasetIntegrityError("legacy users have conflicting community labels")

    refreshes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace["action"] == "refresh":
            refreshes[int(trace["user_id"])].append(trace)
    previous_action_rowid: dict[int, int] = defaultdict(int)
    history: dict[int, list[str]] = defaultdict(list)
    created_derivatives: set[int] = set()
    prior_reports: set[tuple[int, int]] = set()
    used_refreshes: set[int] = set()
    unavailable: list[dict[str, Any]] = []
    examples: list[BehaviorExample] = []
    rationale_report: list[dict[str, Any]] = []
    seen_teacher_ids: set[str] = set()
    decision_sequence = 0

    for trace in traces:
        rowid = int(trace["rowid"])
        user_id = int(trace["user_id"])
        action_name = str(trace["action"])
        token = f"trace-{rowid}:{action_name}"
        if action_name not in ACTION_MAP:
            history[user_id].append(token)
            continue
        decision_sequence += 1
        source_id = f"trace-{rowid}"
        source = teacher_source.get(source_id)
        timestep = int(trace["created_at"])
        candidates = [
            row
            for row in refreshes[user_id]
            if previous_action_rowid[user_id] < int(row["rowid"]) < rowid
            and int(row["created_at"]) == timestep
            and int(row["rowid"]) not in used_refreshes
        ]
        if len(candidates) != 1 or source is None:
            unavailable.append(
                {
                    "candidate_refresh_trace_rowids": sorted(
                        int(row["rowid"]) for row in candidates
                    ),
                    "reason": (
                        "teacher_sample_missing"
                        if source is None
                        else "legacy_refresh_match_is_not_unique"
                    ),
                    "source_sample_id": source_id,
                    "trace_rowid": rowid,
                }
            )
            previous_action_rowid[user_id] = rowid
            history[user_id].append(token)
            continue
        refresh = candidates[0]
        refresh_rowid = int(refresh["rowid"])
        used_refreshes.add(refresh_rowid)
        feed = _json_object(str(refresh["info"]), field="refresh.info")
        visible_ids = _feed_post_ids(feed)
        legal_choices = _legacy_legal_choices(
            user_id=user_id,
            visible_ids=visible_ids,
            posts=posts,
            created_derivatives=created_derivatives,
            reports=prior_reports,
        )
        trace_info = _json_object(str(trace["info"]), field="action.info")
        selected = _trace_choice(action_name, trace_info)
        action = _choice_action(selected)
        target = _choice_target(selected)
        errors: list[str] = []
        if selected not in legal_choices:
            errors.append("recorded choice is absent from reconstructed legal choices")
        if target is not None and target not in visible_ids:
            errors.append("recorded target is absent from reconstructed feed")
        label = source.get("label")
        if not isinstance(label, dict) or label.get("action") != action:
            errors.append("teacher action label disagrees with OASIS trace")
        history_before = list(history[user_id][-10:])
        try:
            _validate_history(history_before, rowid)
        except DatasetIntegrityError as exc:
            errors.append(str(exc))
        if errors:
            unavailable.append(
                {
                    "candidate_refresh_trace_rowids": [refresh_rowid],
                    "reason": "; ".join(errors),
                    "source_sample_id": source_id,
                    "trace_rowid": rowid,
                }
            )
            previous_action_rowid[user_id] = rowid
            history[user_id].append(token)
            continue

        quality_status = rationale_issues.get(source_id, "not_individually_verified")
        rationale_eligible = source_id not in rationale_issues
        original_reason = str(label.get("reason", ""))
        if not rationale_eligible:
            rationale_report.append(
                {
                    "original_rationale": original_reason,
                    "rationale_quality_status": quality_status,
                    "source_sample_id": source_id,
                }
            )
        empty_reason = None
        if not visible_ids:
            empty_reason = str(feed.get("message") or feed.get("error") or "") or None
            if empty_reason is None:
                unavailable.append(
                    {
                        "candidate_refresh_trace_rowids": [refresh_rowid],
                        "reason": "empty feed has no explicit reason",
                        "source_sample_id": source_id,
                        "trace_rowid": rowid,
                    }
                )
                previous_action_rowid[user_id] = rowid
                history[user_id].append(token)
                continue
        decision_id = (
            f"{run_ids[0]}:legacy:t{timestep:06d}:u{user_id:06d}:"
            f"d{decision_sequence:06d}"
        )
        example = BehaviorExample(
            sample_id=decision_id,
            user_profile=str(source.get("user_profile", "")),
            community=str(
                source.get(
                    "community",
                    next(iter(community_values.get(user_id, {"unknown"}))),
                )
            ),
            feed_post=canonical_json(feed),
            neighbor_interactions=[
                str(value) for value in source.get("neighbor_interactions", [])
            ],
            behavior_history=history_before,
            platform_notice=str(source.get("platform_notice", "")),
            label=ActionOutput(
                action=action,
                confidence=float(label.get("confidence", 1.0)),
                reason=original_reason if rationale_eligible else "",
            ),
            label_source=label_source,
            action_trace_rowid=rowid,
            decision_sequence=decision_sequence,
            decision_timestep=timestep,
            visible_post_ids=visible_ids,
            legal_choice_ids=legal_choices,
            selected_choice_id=selected,
            provenance="legacy_refresh_trace_reconstruction",
            action_label_eligible=True,
            rationale_training_eligible=rationale_eligible,
            rationale_quality_status=quality_status,
            feed_empty_reason=empty_reason,
        )
        examples.append(example)
        seen_teacher_ids.add(source_id)
        if action == "report" and target is not None:
            prior_reports.add((user_id, target))
        if action in {"repost", "quote"} and trace_info.get("new_post_id") is not None:
            created_derivatives.add(int(trace_info["new_post_id"]))
        previous_action_rowid[user_id] = rowid
        history[user_id].append(token)

    extra_teacher_ids = sorted(set(teacher_source) - seen_teacher_ids)
    if extra_teacher_ids:
        unavailable.extend(
            {
                "candidate_refresh_trace_rowids": [],
                "reason": "teacher sample has no uniquely reconstructed decision",
                "source_sample_id": sample_id,
                "trace_rowid": None,
            }
            for sample_id in extra_teacher_ids
        )
    examples.sort(key=lambda value: (value.decision_sequence or -1, value.sample_id))
    serialized = serialize_jsonl(examples)
    sensitive = bool(SECRET_PATTERN.search(serialized))
    report = {
        "action_label_eligible": sum(e.action_label_eligible for e in examples),
        "ambiguous_or_unavailable": sorted(
            unavailable,
            key=lambda value: (
                value["trace_rowid"] is None,
                value["trace_rowid"] or 0,
                value["source_sample_id"],
            ),
        ),
        "input_hashes": {
            db_path.name: _file_sha256(db_path),
            teacher_source_path.name: _file_sha256(teacher_source_path),
            **(
                {quality_audit_path.name: _file_sha256(quality_audit_path)}
                if quality_audit_path is not None
                else {}
            ),
        },
        "matched_decisions": len(examples),
        "output_examples": len(examples),
        "output_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "provenance": "legacy_refresh_trace_reconstruction",
        "rationale_quality_exclusions": rationale_report,
        "rationale_training_eligible": sum(
            e.rationale_training_eligible for e in examples
        ),
        "secret_pattern_scan_passed": not sensitive,
        "source_action_traces": sum(trace["action"] in ACTION_MAP for trace in traces),
        "status": "success" if not unavailable and not sensitive else "failed",
        "unique_refresh_matches": len(used_refreshes),
        "warning": (
            "Legacy contexts were reconstructed from explicitly paired refresh and "
            "action traces; they are not equivalent to native DecisionSnapshot rows."
        ),
    }
    return DatasetBuild(examples=tuple(examples), report=report)


def serialize_jsonl(
    examples: list[BehaviorExample] | tuple[BehaviorExample, ...],
) -> str:
    ordered = sorted(
        examples,
        key=lambda value: (
            value.decision_sequence is None,
            value.decision_sequence if value.decision_sequence is not None else 0,
            value.sample_id,
        ),
    )
    return "".join(canonical_json(example.to_dict()) + "\n" for example in ordered)


def write_dataset(examples: tuple[BehaviorExample, ...], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_jsonl(examples), encoding="utf-8")
    return len(examples)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DatasetBuild",
    "DatasetIntegrityError",
    "build_from_decision_snapshots",
    "has_decision_snapshots",
    "reconstruct_legacy_dataset",
    "serialize_jsonl",
    "write_dataset",
    "write_report",
]
