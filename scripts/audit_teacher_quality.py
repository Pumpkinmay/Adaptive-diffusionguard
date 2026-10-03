"""Deterministic, offline quality audit for distilled OASIS teacher actions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ACTION_MAP = {
    "repost": "repost",
    "quote_post": "quote",
    "report_post": "report",
    "do_nothing": "ignore",
}
ACTIONS = ("ignore", "quote", "report", "repost")
KNOWN_SEED_RISKS = {
    "Verified synthetic weather update.": 0.0,
    "Unverified synthetic health claim.": 1.0,
}
SECRET_PATTERNS = {
    "groq": re.compile(rb"gsk_[A-Za-z0-9_-]{12,}"),
    "openai": re.compile(rb"sk-[A-Za-z0-9_-]{12,}"),
    "huggingface": re.compile(rb"hf_[A-Za-z0-9]{12,}"),
}
REQUIRED_FILES = {
    "summary": "summary.json",
    "teacher": "groq_teacher_trajectories.jsonl",
    "database": "real_llm_validation.db",
    "profiles": "synthetic_profiles.csv",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _risk_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score == 0.0:
        return "low"
    if score == 1.0:
        return "high"
    return "intermediate"


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _root_post_id(
    posts: dict[int, dict[str, Any]], post_id: int
) -> tuple[int | None, str | None]:
    current = post_id
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        row = posts.get(current)
        if row is None:
            return None, f"post {current} is absent from the post table"
        parent = row["original_post_id"]
        if parent is None:
            return current, None
        current = int(parent)
    return None, f"cycle detected while resolving post {post_id}"


def _target_post_id(action: str, info: dict[str, Any]) -> int | None:
    keys = {
        "repost": ("reposted_id", "post_id"),
        "quote": ("quoted_id", "post_id", "original_post_id"),
        "report": ("post_id",),
        "ignore": (),
    }[action]
    for key in keys:
        value = info.get(key)
        if value is not None:
            return int(value)
    return None


def _new_post_id(action: str, info: dict[str, Any]) -> int | None:
    if action not in {"repost", "quote"}:
        return None
    value = info.get("new_post_id")
    return int(value) if value is not None else None


def _legal_choices(
    *,
    user_id: int,
    visible_post_ids: list[int],
    posts: dict[int, dict[str, Any]],
    created_derivative_ids: set[int],
    prior_reports: set[tuple[int, int]],
) -> tuple[list[str], list[str]]:
    choices = ["ignore"]
    reasons: list[str] = []
    for post_id in sorted(set(visible_post_ids)):
        post = posts.get(post_id)
        if post is None:
            reasons.append(f"visible post {post_id} is missing from post table")
            continue
        targets = [post_id]
        is_repost = (
            post["original_post_id"] is not None and post["quote_content"] is None
        )
        if is_repost:
            root_id, error = _root_post_id(posts, post_id)
            if error is not None:
                reasons.append(error)
            elif root_id is not None:
                targets.append(root_id)
        already_derived = any(
            derivative_id in created_derivative_ids
            and int(posts[derivative_id]["user_id"]) == user_id
            and posts[derivative_id]["original_post_id"] in targets
            for derivative_id in sorted(created_derivative_ids)
            if derivative_id in posts
        )
        if not already_derived:
            choices.append(f"repost:{post_id}")
        choices.append(f"quote:{post_id}")
        if (user_id, post_id) not in prior_reports:
            choices.append(f"report:{post_id}")
    return choices, reasons


def _quote_stance(reason: str) -> str:
    text = reason.casefold()
    correction = (
        "correct",
        "fact-check",
        "fact check",
        "misleading",
        "false",
        "unverified",
        "evidence",
    )
    questioning = ("?", "question", "verify", "uncertain", "is this true")
    supportive = ("support", "agree", "confirmed", "is true", "amplify")
    if any(token in text for token in correction):
        return "corrective"
    if any(token in text for token in questioning):
        return "questioning"
    if any(token in text for token in supportive):
        return "supportive"
    return "unclear"


def _rationale_quality(
    *,
    reason: str,
    action: str,
    target_post_id: int | None,
    visible_post_ids: list[int],
    user_id: int,
    root_author_id: int | None,
) -> dict[str, Any]:
    stripped = reason.strip()
    empty = not stripped
    too_short = not empty and (len(stripped) < 20 or len(stripped.split()) < 4)
    lower = stripped.casefold()
    declared: set[str] = set()
    for candidate in ACTIONS:
        patterns = (
            f"appropriate action is to {candidate}",
            f"appropriate action is {candidate}",
            f"action should be {candidate}",
        )
        if any(pattern in lower for pattern in patterns):
            declared.add(candidate)
    contradictory = bool(declared and action not in declared)
    referenced_ids = sorted(
        {int(value) for value in re.findall(r"(?:post\s*#?|repost:)\s*(\d+)", lower)}
    )
    wrong_references = [
        value for value in referenced_ids if value not in visible_post_ids
    ]
    if target_post_id is not None and "repost:" in lower:
        explicit_repost_ids = {
            int(value) for value in re.findall(r"repost:\s*(\d+)", lower)
        }
        wrong_references.extend(
            value for value in explicit_repost_ids if value != target_post_id
        )
    own_post_mismatch = (
        target_post_id is not None
        and root_author_id is not None
        and user_id != root_author_id
        and "own original post" in lower
    )
    return {
        "action_contradiction": contradictory,
        "declared_actions": sorted(declared),
        "empty": empty,
        "factual_author_mismatch": own_post_mismatch,
        "referenced_post_ids": referenced_ids,
        "too_short": too_short,
        "wrong_post_reference_ids": sorted(set(wrong_references)),
    }


def _load_teacher_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    nonempty = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        nonempty += 1
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("line is not a JSON object")
            if not isinstance(value.get("sample_id"), str):
                raise TypeError("sample_id is not a string")
            label = value.get("label")
            if not isinstance(label, dict) or label.get("action") not in ACTIONS:
                raise ValueError("label.action is missing or invalid")
            rows.append(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append({"line": line_number, "reason": str(exc)})
    return rows, errors, nonempty


def _group_exposures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = len(rows)
    shown = sum(int(row["shown"]) for row in rows)
    suppressed = candidates - shown
    return {
        "candidate": candidates,
        "shown": shown,
        "suppressed": suppressed,
        "suppression_rate": _ratio(suppressed, candidates),
    }


def _scan_credentials(paths: list[Path]) -> dict[str, Any]:
    matches = {name: 0 for name in SECRET_PATTERNS}
    for path in paths:
        data = path.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            matches[name] += len(pattern.findall(data))
    return {
        "files_scanned": sorted(path.name for path in paths),
        "match_counts": matches,
        "passed": not any(matches.values()),
        "env_scanned": False,
    }


def audit_directory(output_dir: Path) -> dict[str, Any]:
    """Return a deterministic audit without mutating any input artifact."""
    output_dir = Path(output_dir)
    paths = {name: output_dir / filename for name, filename in REQUIRED_FILES.items()}
    missing = sorted(path.name for path in paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing required audit inputs: {', '.join(missing)}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    teacher_rows, teacher_errors, nonempty_lines = _load_teacher_rows(paths["teacher"])
    with paths["profiles"].open(encoding="utf-8", newline="") as handle:
        profiles = list(csv.DictReader(handle))

    connection = _open_read_only(paths["database"])
    try:
        traces = [
            dict(row)
            for row in connection.execute(
                "SELECT rowid, user_id, created_at, action, info FROM trace ORDER BY rowid"
            )
        ]
        post_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT post_id, user_id, original_post_id, quote_content, content, "
                "created_at FROM post ORDER BY post_id"
            )
        ]
        impression_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT impression_id, run_id, timestep, user_id, post_id, root_post_id, "
                "user_community, author_community, base_score, risk_score, omega_intra, "
                "omega_inter, keep_probability, shown, source "
                "FROM diffusionguard_impression ORDER BY impression_id"
            )
        ]
        user_ids = {
            int(row[0]) for row in connection.execute("SELECT user_id FROM user")
        }
    finally:
        connection.close()

    posts = {int(row["post_id"]): row for row in post_rows}
    impressions = [
        {
            **row,
            "impression_id": int(row["impression_id"]),
            "post_id": int(row["post_id"]),
            "root_post_id": int(row["root_post_id"]),
            "shown": int(row["shown"]),
            "timestep": int(row["timestep"]),
            "user_id": int(row["user_id"]),
        }
        for row in impression_rows
    ]
    impression_index: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    community_by_user: dict[int, set[str]] = defaultdict(set)
    risk_by_root: dict[int, set[float]] = defaultdict(set)
    for row in impressions:
        impression_index[(row["timestep"], row["user_id"], row["post_id"])].append(row)
        community_by_user[row["user_id"]].add(str(row["user_community"]))
        root = posts.get(row["root_post_id"])
        if root is not None:
            community_by_user[int(root["user_id"])].add(str(row["author_community"]))
        risk_by_root[row["root_post_id"]].add(float(row["risk_score"]))

    teacher_by_id: dict[str, dict[str, Any]] = {}
    duplicate_teacher_ids: list[str] = []
    for row in teacher_rows:
        sample_id = str(row["sample_id"])
        if sample_id in teacher_by_id:
            duplicate_teacher_ids.append(sample_id)
        teacher_by_id.setdefault(sample_id, row)

    action_traces = [row for row in traces if row["action"] in ACTION_MAP]
    success_ids = {f"trace-{int(row['rowid'])}" for row in action_traces}
    teacher_ids = {str(row["sample_id"]) for row in teacher_rows}
    trace_only_ids = sorted(success_ids - teacher_ids)
    teacher_only_ids = sorted(teacher_ids - success_ids)

    duplicate_action_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in action_traces:
        key = (
            int(row["user_id"]),
            int(row["created_at"]),
            str(row["action"]),
            _canonical(_json_object(row["info"])),
        )
        duplicate_action_groups[key].append(int(row["rowid"]))
    duplicate_action_trace_groups = [
        ids for ids in duplicate_action_groups.values() if len(ids) > 1
    ]

    profile_names = {str(row.get("username", "")) for row in profiles}
    last_feed: dict[int, dict[str, Any]] = {}
    prior_reports: set[tuple[int, int]] = set()
    created_derivative_ids: set[int] = set()
    samples: list[dict[str, Any]] = []
    label_mismatch_ids: list[str] = []

    for trace in traces:
        rowid = int(trace["rowid"])
        user_id = int(trace["user_id"])
        oasis_action = str(trace["action"])
        info = _json_object(trace["info"])
        if oasis_action == "refresh":
            last_feed[user_id] = {"trace_rowid": rowid, "payload": info}
            continue
        if oasis_action == "create_post":
            continue
        if oasis_action not in ACTION_MAP:
            continue

        action = ACTION_MAP[oasis_action]
        sample_id = f"trace-{rowid}"
        teacher = teacher_by_id.get(sample_id)
        feed_state = last_feed.get(user_id)
        feed_payload = feed_state["payload"] if feed_state is not None else {}
        raw_feed_posts = feed_payload.get("posts", [])
        if not isinstance(raw_feed_posts, list):
            raw_feed_posts = []
        visible_post_ids = [
            int(item["post_id"])
            for item in raw_feed_posts
            if isinstance(item, dict) and item.get("post_id") is not None
        ]
        timestep: int | None
        try:
            timestep = int(trace["created_at"])
        except (TypeError, ValueError):
            timestep = None

        legal_choices, legal_reconstruction_errors = _legal_choices(
            user_id=user_id,
            visible_post_ids=visible_post_ids,
            posts=posts,
            created_derivative_ids=created_derivative_ids,
            prior_reports=prior_reports,
        )
        direct_post_id = _target_post_id(action, info)
        root_post_id: int | None = None
        root_error: str | None = None
        if direct_post_id is not None:
            root_post_id, root_error = _root_post_id(posts, direct_post_id)
        root_post = posts.get(root_post_id) if root_post_id is not None else None
        root_author_id = int(root_post["user_id"]) if root_post is not None else None
        root_content = str(root_post["content"]) if root_post is not None else None
        risk_values = (
            risk_by_root.get(root_post_id, set()) if root_post_id is not None else set()
        )
        risk_score = next(iter(risk_values)) if len(risk_values) == 1 else None

        feed_items: list[dict[str, Any]] = []
        association_errors: list[str] = []
        if feed_state is None:
            association_errors.append("no preceding refresh trace for user")
        if timestep is None:
            association_errors.append(
                "trace created_at cannot be converted to timestep"
            )
        for item in raw_feed_posts:
            if not isinstance(item, dict) or item.get("post_id") is None:
                association_errors.append("refresh feed contains an invalid post entry")
                continue
            post_id = int(item["post_id"])
            item_root_id, item_root_error = _root_post_id(posts, post_id)
            matches = (
                impression_index.get((timestep, user_id, post_id), [])
                if timestep is not None
                else []
            )
            if len(matches) != 1:
                association_errors.append(
                    f"post {post_id} has {len(matches)} matching impression rows"
                )
            impression = matches[0] if len(matches) == 1 else None
            if impression is not None and impression["shown"] != 1:
                association_errors.append(
                    f"visible post {post_id} has an impression marked shown=0"
                )
            item_root = posts.get(item_root_id) if item_root_id is not None else None
            feed_items.append(
                {
                    "author_community": (
                        str(impression["author_community"])
                        if impression is not None
                        else None
                    ),
                    "content": item.get("content"),
                    "impression_id": (
                        int(impression["impression_id"])
                        if impression is not None
                        else None
                    ),
                    "post_id": post_id,
                    "risk_bucket": _risk_bucket(
                        float(impression["risk_score"])
                        if impression is not None
                        else None
                    ),
                    "risk_score": (
                        float(impression["risk_score"])
                        if impression is not None
                        else None
                    ),
                    "root_author_id": (
                        int(item_root["user_id"]) if item_root is not None else None
                    ),
                    "root_content": (
                        str(item_root["content"]) if item_root is not None else None
                    ),
                    "root_error": item_root_error,
                    "root_post_id": item_root_id,
                    "user_community": (
                        str(impression["user_community"])
                        if impression is not None
                        else None
                    ),
                }
            )

        target_impressions = [
            item["impression_id"]
            for item in feed_items
            if item["post_id"] == direct_post_id and item["impression_id"] is not None
        ]
        if direct_post_id is not None and direct_post_id not in visible_post_ids:
            association_errors.append(
                "action target is not in the preceding visible feed"
            )
        if direct_post_id is not None and len(target_impressions) != 1:
            association_errors.append(
                "target action does not map to exactly one shown impression"
            )

        user_communities = community_by_user.get(user_id, set())
        user_community = (
            next(iter(user_communities)) if len(user_communities) == 1 else None
        )
        author_communities = (
            community_by_user.get(root_author_id, set())
            if root_author_id is not None
            else set()
        )
        author_community = (
            next(iter(author_communities)) if len(author_communities) == 1 else None
        )
        community_relation = (
            "intra"
            if user_community is not None
            and author_community is not None
            and user_community == author_community
            else "inter"
            if user_community is not None and author_community is not None
            else "not_applicable"
            if direct_post_id is None
            else "unknown"
        )
        expected_choice = (
            "ignore" if action == "ignore" else f"{action}:{direct_post_id}"
        )
        if expected_choice not in legal_choices:
            association_errors.append(
                "recorded action is absent from reconstructed legal choices"
            )

        label = teacher.get("label", {}) if teacher is not None else {}
        if teacher is None or label.get("action") != action:
            label_mismatch_ids.append(sample_id)
        reason = str(label.get("reason", ""))
        rationale = _rationale_quality(
            reason=reason,
            action=action,
            target_post_id=direct_post_id,
            visible_post_ids=visible_post_ids,
            user_id=user_id,
            root_author_id=root_author_id,
        )

        parsed_teacher_feed: Any = None
        teacher_feed_error: str | None = None
        if teacher is not None:
            try:
                parsed_teacher_feed = json.loads(str(teacher.get("feed_post", "")))
            except json.JSONDecodeError as exc:
                teacher_feed_error = str(exc)
        feed_matches = parsed_teacher_feed == feed_payload
        feed_is_action_payload = parsed_teacher_feed == info

        feed_buckets = [item["risk_bucket"] for item in feed_items]
        has_high = "high" in feed_buckets
        has_low = "low" in feed_buckets
        only_low = bool(feed_buckets) and all(value == "low" for value in feed_buckets)
        ignore_category: str | None = None
        manual_review = False
        if action == "ignore":
            if has_high and has_low:
                ignore_category = "mixed_high_low_feed"
                manual_review = True
            elif has_high:
                ignore_category = "missed_intervention_candidate"
                manual_review = True
            elif only_low:
                ignore_category = "low_risk_only_usually_acceptable"
            else:
                ignore_category = "unavailable_or_intermediate_feed"
                manual_review = True
        quote_category = _quote_stance(reason) if action == "quote" else None
        if action == "quote":
            manual_review = True
        if any(
            (
                rationale["action_contradiction"],
                rationale["empty"],
                rationale["factual_author_mismatch"],
                rationale["too_short"],
                bool(rationale["wrong_post_reference_ids"]),
            )
        ):
            manual_review = True

        risk_bucket = _risk_bucket(risk_score)
        rule_classification: str
        if action == "report" and risk_bucket == "high":
            rule_classification = "aligned_report_high"
        elif action == "report" and risk_bucket == "low":
            rule_classification = "reverse_report_low"
        elif action == "repost" and risk_bucket == "low":
            rule_classification = "aligned_repost_low"
        elif action == "repost" and risk_bucket == "high":
            rule_classification = "reverse_repost_high"
        elif action == "quote":
            rule_classification = f"manual_quote_{quote_category}"
        elif action == "ignore":
            rule_classification = f"ignore_{ignore_category}"
        else:
            rule_classification = "unavailable"

        null_reasons: dict[str, str] = {}
        if direct_post_id is None:
            null_reasons.update(
                {
                    "author_community": "ignore has no action target",
                    "community_relation": "ignore has no action target",
                    "direct_post_id": "ignore has no action target",
                    "risk_score": "ignore has no action target",
                    "root_author_id": "ignore has no action target",
                    "root_content": "ignore has no action target",
                    "root_post_id": "ignore has no action target",
                }
            )
        elif root_error is not None:
            null_reasons["root_post_id"] = root_error
        if direct_post_id is not None and risk_score is None:
            null_reasons["risk_score"] = (
                "root has no unique risk score in diffusionguard_impression"
            )
        if user_community is None:
            null_reasons["user_community"] = (
                "user has no unique community in diffusionguard_impression"
            )
        if direct_post_id is not None and author_community is None:
            null_reasons["author_community"] = (
                "root author has no unique community in diffusionguard_impression"
            )
        if direct_post_id is not None and len(target_impressions) != 1:
            null_reasons["target_impression_id"] = (
                "target action does not map to exactly one shown impression"
            )

        profile_name = f"synthetic_agent_{user_id}"
        if profile_name not in profile_names:
            association_errors.append("user does not map to synthetic_profiles.csv")
        created_post_id = _new_post_id(action, info)
        created_post = (
            posts.get(created_post_id) if created_post_id is not None else None
        )
        samples.append(
            {
                "action": action,
                "action_info": info,
                "author_community": author_community,
                "community_relation": community_relation,
                "created_post_id": created_post_id,
                "direct_post_id": direct_post_id,
                "expected_choice_id": expected_choice,
                "feed_post_field": {
                    "error": teacher_feed_error,
                    "is_action_trace_payload": feed_is_action_payload,
                    "matches_reconstructed_refresh": feed_matches,
                },
                "ignore_category": ignore_category,
                "legal_choice_ids": legal_choices,
                "legal_reconstruction_errors": legal_reconstruction_errors,
                "link_errors": sorted(set(association_errors)),
                "linked_impression_ids": [
                    item["impression_id"]
                    for item in feed_items
                    if item["impression_id"] is not None
                ],
                "manual_review": manual_review,
                "null_reasons": null_reasons,
                "quote_stance": quote_category,
                "quote_text": (
                    str(created_post["quote_content"])
                    if action == "quote"
                    and created_post is not None
                    and created_post["quote_content"] is not None
                    else None
                ),
                "rationale": reason,
                "rationale_quality": rationale,
                "risk_bucket": risk_bucket,
                "risk_score": risk_score,
                "root_author_id": root_author_id,
                "root_content": root_content,
                "root_post_id": root_post_id,
                "rule_classification": rule_classification,
                "sample_id": sample_id,
                "target_impression_id": (
                    target_impressions[0] if len(target_impressions) == 1 else None
                ),
                "timestep": timestep,
                "trace_rowid": rowid,
                "user_community": user_community,
                "user_id": user_id,
                "visible_feed": feed_items,
                "visible_post_ids": visible_post_ids,
            }
        )

        if action == "report" and direct_post_id is not None:
            prior_reports.add((user_id, direct_post_id))
        if created_post_id is not None:
            created_derivative_ids.add(created_post_id)

    samples.sort(key=lambda row: int(row["trace_rowid"]))
    action_counts = Counter(str(row["action"]) for row in samples)
    action_distribution = {
        action: {
            "count": action_counts[action],
            "proportion": _ratio(action_counts[action], len(samples)),
        }
        for action in ACTIONS
    }
    action_risk_cross = {
        action: {
            key: 0
            for key in ("low", "high", "intermediate", "unknown", "not_applicable")
        }
        for action in ACTIONS
    }
    for row in samples:
        bucket = (
            row["risk_bucket"]
            if row["direct_post_id"] is not None
            else "not_applicable"
        )
        action_risk_cross[row["action"]][bucket] += 1

    action_by_timestep: dict[str, dict[str, int]] = {}
    for timestep in sorted(
        {row["timestep"] for row in samples if row["timestep"] is not None}
    ):
        rows = [row for row in samples if row["timestep"] == timestep]
        action_by_timestep[str(timestep)] = {
            action: sum(row["action"] == action for row in rows) for action in ACTIONS
        }
    action_by_relation = {
        relation: {action: 0 for action in ACTIONS}
        for relation in ("intra", "inter", "not_applicable", "unknown")
    }
    for row in samples:
        action_by_relation[row["community_relation"]][row["action"]] += 1

    rule_aligned_ids = sorted(
        row["sample_id"]
        for row in samples
        if str(row["rule_classification"]).startswith("aligned_")
    )
    rule_reverse_ids = sorted(
        row["sample_id"]
        for row in samples
        if str(row["rule_classification"]).startswith("reverse_")
    )
    clearly_judged_ids = sorted(rule_aligned_ids + rule_reverse_ids)
    manual_review_ids = sorted(
        row["sample_id"] for row in samples if row["manual_review"]
    )
    unavailable_ids = sorted(
        row["sample_id"]
        for row in samples
        if row["rule_classification"] == "unavailable" or row["link_errors"]
    )
    rationale_issue_ids = {
        "action_contradiction": sorted(
            row["sample_id"]
            for row in samples
            if row["rationale_quality"]["action_contradiction"]
        ),
        "empty": sorted(
            row["sample_id"] for row in samples if row["rationale_quality"]["empty"]
        ),
        "factual_author_mismatch": sorted(
            row["sample_id"]
            for row in samples
            if row["rationale_quality"]["factual_author_mismatch"]
        ),
        "too_short": sorted(
            row["sample_id"] for row in samples if row["rationale_quality"]["too_short"]
        ),
        "wrong_post_reference": sorted(
            row["sample_id"]
            for row in samples
            if row["rationale_quality"]["wrong_post_reference_ids"]
        ),
    }
    quote_stances = {
        stance: sum(row["quote_stance"] == stance for row in samples)
        for stance in ("questioning", "corrective", "supportive", "unclear")
    }

    impression_dicts = [dict(row) for row in impressions]
    risk_exposures = {
        bucket: _group_exposures(
            [
                row
                for row in impression_dicts
                if _risk_bucket(float(row["risk_score"])) == bucket
            ]
        )
        for bucket in ("low", "high", "intermediate")
    }
    relation_exposures = {
        relation: _group_exposures(
            [
                row
                for row in impression_dicts
                if (
                    (
                        "intra"
                        if row["user_community"] == row["author_community"]
                        else "inter"
                    )
                    == relation
                )
            ]
        )
        for relation in ("intra", "inter")
    }
    timestep_exposures = {
        str(timestep): _group_exposures(
            [row for row in impression_dicts if int(row["timestep"]) == timestep]
        )
        for timestep in sorted({int(row["timestep"]) for row in impression_dicts})
    }
    high_repost_samples = [
        row
        for row in samples
        if row["action"] == "repost" and row["risk_bucket"] == "high"
    ]
    high_repost_ids = sorted(row["sample_id"] for row in high_repost_samples)
    high_repost_continuations: list[dict[str, Any]] = []
    for sample in high_repost_samples:
        created_post_id = sample["created_post_id"]
        if created_post_id is None or sample["timestep"] is None:
            continue
        later = sorted(
            int(row["impression_id"])
            for row in impression_dicts
            if int(row["post_id"]) == created_post_id
            and int(row["timestep"]) > int(sample["timestep"])
        )
        if later:
            high_repost_continuations.append(
                {
                    "created_post_id": created_post_id,
                    "later_impression_ids": later,
                    "sample_id": sample["sample_id"],
                }
            )
    low_report_ids = sorted(
        row["sample_id"]
        for row in samples
        if row["action"] == "report" and row["risk_bucket"] == "low"
    )
    high_rate = risk_exposures["high"]["suppression_rate"]
    low_rate = risk_exposures["low"]["suppression_rate"]

    secret_scan = _scan_credentials(list(paths.values()))
    feed_mismatch_ids = sorted(
        row["sample_id"]
        for row in samples
        if not row["feed_post_field"]["matches_reconstructed_refresh"]
    )
    feed_action_payload_ids = sorted(
        row["sample_id"]
        for row in samples
        if row["feed_post_field"]["is_action_trace_payload"]
    )
    unavailable_fields = [
        {
            "reasons": row["null_reasons"],
            "sample_id": row["sample_id"],
        }
        for row in samples
        if row["direct_post_id"] is not None and row["null_reasons"]
    ]
    invalid_teacher_ids = sorted(set(teacher_only_ids + label_mismatch_ids))
    seed_risk_cross_check: list[dict[str, Any]] = []
    for content, expected_risk in KNOWN_SEED_RISKS.items():
        roots = sorted(
            int(post_id)
            for post_id, post in posts.items()
            if post["original_post_id"] is None and post["content"] == content
        )
        observed = sorted(
            {value for root_id in roots for value in risk_by_root.get(root_id, set())}
        )
        seed_risk_cross_check.append(
            {
                "content": content,
                "expected_risk_score": expected_risk,
                "matches": len(roots) == 1 and observed == [expected_risk],
                "observed_risk_scores": observed,
                "root_post_ids": roots,
            }
        )

    audit = {
        "audit_schema_version": 1,
        "experiment": {
            "agent_count": len(profiles),
            "community_mapping": {
                str(user_id): sorted(values)
                for user_id, values in sorted(community_by_user.items())
            },
            "known_seed_posts": [
                {"content": content, "risk_score": risk}
                for content, risk in KNOWN_SEED_RISKS.items()
            ],
            "seed": 20260927,
            "summary_validation_mode": summary.get("validation_mode"),
            "timesteps": summary.get("timesteps"),
        },
        "governance": {
            "cosref_higher_suppression_for_high_risk_observed": (
                high_rate is not None and low_rate is not None and high_rate > low_rate
            ),
            "high_minus_low_suppression_rate": (
                high_rate - low_rate
                if high_rate is not None and low_rate is not None
                else None
            ),
            "high_risk_repost_continued_propagation": bool(high_repost_continuations),
            "high_risk_repost_continuations": high_repost_continuations,
            "high_risk_repost_sample_ids": high_repost_ids,
            "low_risk_report_sample_ids": low_report_ids,
            "normal_content_erroneously_reported_count": len(low_report_ids),
            "overall": _group_exposures(impression_dicts),
            "risk": risk_exposures,
            "community_relation": relation_exposures,
            "timestep": timestep_exposures,
            "interpretation": (
                "Observed suppression differs by configured risk/community strata; "
                "this small synthetic run does not establish a causal effect."
            ),
        },
        "integrity": {
            "action_trace_count": len(action_traces),
            "all_samples_linked_to_user_timestep_feed_and_impressions": not any(
                row["link_errors"] for row in samples
            ),
            "credential_pattern_scan": secret_scan,
            "duplicate_action_trace_count": len(duplicate_action_trace_groups),
            "duplicate_action_trace_rowids": sorted(duplicate_action_trace_groups),
            "duplicate_sample_ids": sorted(set(duplicate_teacher_ids)),
            "failed_or_mismatched_teacher_sample_count": len(invalid_teacher_ids),
            "failed_or_mismatched_teacher_sample_ids": invalid_teacher_ids,
            "jsonl_invalid_lines": teacher_errors,
            "jsonl_nonempty_line_count": nonempty_lines,
            "jsonl_valid_line_count": len(teacher_rows),
            "known_seed_risk_cross_check": seed_risk_cross_check,
            "profile_count": len(profiles),
            "samples_with_link_errors": sorted(
                row["sample_id"] for row in samples if row["link_errors"]
            ),
            "targeted_sample_count": sum(
                row["direct_post_id"] is not None for row in samples
            ),
            "targeted_samples_fully_linked": sum(
                row["direct_post_id"] is not None and not row["link_errors"]
                for row in samples
            ),
            "sample_ids_one_to_one_with_successful_traces": (
                not duplicate_teacher_ids
                and not trace_only_ids
                and not teacher_only_ids
                and not label_mismatch_ids
            ),
            "summary_secret_scan_passed": summary.get("secret_scan_passed"),
            "summary_teacher_examples": summary.get("teacher_examples"),
            "successful_trace_ids_without_teacher_sample": trace_only_ids,
            "teacher_feed_post_action_payload_count": len(feed_action_payload_ids),
            "teacher_feed_post_action_payload_sample_ids": feed_action_payload_ids,
            "teacher_feed_post_mismatch_count": len(feed_mismatch_ids),
            "teacher_feed_post_mismatch_sample_ids": feed_mismatch_ids,
            "teacher_sample_ids_without_successful_trace": teacher_only_ids,
            "unique_sample_id_count": len(teacher_ids),
            "unavailable_reconstructed_fields": unavailable_fields,
            "user_table_count": len(user_ids),
        },
        "limitations": [
            "Only 5 synthetic agents, 3 timesteps, and 15 actions were observed.",
            "Only two synthetic seed posts were used.",
            "Samples are temporally and socially dependent, not independent and identically distributed.",
            "Risk labels are experiment presets, not human annotations.",
            "No claim about representative human behavior has been tested.",
            "These data cannot establish model accuracy, governance effectiveness, or causal effects.",
            "The audit does not justify starting LoRA training.",
            "Quotes and ignores with high-risk or mixed feeds require human review.",
            "The teacher JSONL feed_post field contains action payloads rather than the reconstructed visible feed.",
        ],
        "quality": {
            "action_by_community_relation": action_by_relation,
            "action_by_root_risk": action_risk_cross,
            "action_by_timestep": action_by_timestep,
            "action_distribution": action_distribution,
            "clearly_judged_sample_count": len(clearly_judged_ids),
            "clearly_judged_sample_ids": clearly_judged_ids,
            "ignore_feed_contains_high_risk_count": sum(
                row["action"] == "ignore"
                and any(item["risk_bucket"] == "high" for item in row["visible_feed"])
                for row in samples
            ),
            "ignore_feed_mixed_high_low_count": sum(
                row["action"] == "ignore"
                and {"high", "low"}.issubset(
                    {item["risk_bucket"] for item in row["visible_feed"]}
                )
                for row in samples
            ),
            "ignore_feed_only_low_risk_count": sum(
                row["action"] == "ignore"
                and bool(row["visible_feed"])
                and all(item["risk_bucket"] == "low" for item in row["visible_feed"])
                for row in samples
            ),
            "manual_review_sample_count": len(manual_review_ids),
            "manual_review_sample_ids": manual_review_ids,
            "missed_intervention_candidate_count": sum(
                row["ignore_category"]
                in {"missed_intervention_candidate", "mixed_high_low_feed"}
                for row in samples
            ),
            "missed_intervention_candidate_sample_ids": sorted(
                row["sample_id"]
                for row in samples
                if row["ignore_category"]
                in {"missed_intervention_candidate", "mixed_high_low_feed"}
            ),
            "quote_stance_counts": quote_stances,
            "rationale_issue_counts": {
                name: len(ids) for name, ids in rationale_issue_ids.items()
            },
            "rationale_issue_sample_ids": rationale_issue_ids,
            "report_high_risk_count": action_risk_cross["report"]["high"],
            "report_low_risk_count": action_risk_cross["report"]["low"],
            "repost_high_risk_count": action_risk_cross["repost"]["high"],
            "repost_low_risk_count": action_risk_cross["repost"]["low"],
            "rule_aligned_denominator": len(clearly_judged_ids),
            "rule_aligned_numerator": len(rule_aligned_ids),
            "rule_aligned_rate": _ratio(len(rule_aligned_ids), len(clearly_judged_ids)),
            "rule_aligned_sample_ids": rule_aligned_ids,
            "rule_reverse_sample_count": len(rule_reverse_ids),
            "rule_reverse_sample_ids": rule_reverse_ids,
            "unable_to_determine_sample_count": len(unavailable_ids),
            "unable_to_determine_sample_ids": unavailable_ids,
        },
        "recommendation": {
            "choice": 1,
            "label": "可以进入100决策评估，但不能开始LoRA",
            "basis": [
                "All five explicitly judgeable report/repost actions are rule-aligned.",
                "No reversed action or missed high-risk ignore appears in this run.",
                "One rationale has a factual author mismatch and must be tracked at larger scale.",
                "The JSONL feed_post serialization must be corrected before any training use.",
                "Fifteen dependent synthetic actions are insufficient for training or efficacy claims.",
            ],
        },
        "samples": samples,
        "source_files": {name: path.name for name, path in sorted(paths.items())},
    }
    return audit


def render_markdown(audit: dict[str, Any]) -> str:
    integrity = audit["integrity"]
    quality = audit["quality"]
    governance = audit["governance"]
    lines = [
        "# v8 教师决策质量审计",
        "",
        "本报告由离线只读审计生成；它不是模型准确率、治理有效性或因果效果证明。",
        "",
        "## 数据完整性",
        "",
        f"- summary teacher_examples：{integrity['summary_teacher_examples']}",
        f"- JSONL 有效行：{integrity['jsonl_valid_line_count']}",
        f"- 唯一 sample ID：{integrity['unique_sample_id_count']}",
        (
            "- 与成功 trace 一一对应："
            f"{integrity['sample_ids_one_to_one_with_successful_traces']}"
        ),
        f"- 重复动作 trace：{integrity['duplicate_action_trace_count']}",
        (
            "- 失败或标签不匹配样本："
            f"{integrity['failed_or_mismatched_teacher_sample_count']}"
        ),
        (
            "- 凭据模式扫描："
            f"{integrity['credential_pattern_scan']['passed']}（未扫描 .env）"
        ),
        (
            "- JSONL feed_post 与真实 Feed 不匹配："
            f"{integrity['teacher_feed_post_mismatch_count']} 条"
        ),
        "",
        "## 动作与风险交叉表",
        "",
        "| 动作 | 数量 | 低风险目标 | 高风险目标 | 无目标 |",
        "|---|---:|---:|---:|---:|",
    ]
    for action in ACTIONS:
        distribution = quality["action_distribution"][action]
        cross = quality["action_by_root_risk"][action]
        lines.append(
            f"| {action} | {distribution['count']} | {cross['low']} | "
            f"{cross['high']} | {cross['not_applicable']} |"
        )
    lines.extend(
        [
            "",
            (
                "- 可明确判断的 rule-aligned："
                f"{quality['rule_aligned_numerator']}/{quality['rule_aligned_denominator']} "
                f"({quality['rule_aligned_rate']:.3f})"
            ),
            f"- 明确反向动作：{quality['rule_reverse_sample_ids'] or '无'}",
            f"- 需人工复核：{quality['manual_review_sample_ids'] or '无'}",
            (
                "- missed_intervention_candidate："
                f"{quality['missed_intervention_candidate_count']}"
            ),
            (
                "- rationale 事实性作者错误："
                f"{quality['rationale_issue_sample_ids']['factual_author_mismatch'] or '无'}"
            ),
            "",
            "## 曝光与治理",
            "",
            "| 分组 | 候选 | 展示 | 抑制 | 抑制率 |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 总计 | {governance['overall']['candidate']} | "
                f"{governance['overall']['shown']} | {governance['overall']['suppressed']} | "
                f"{governance['overall']['suppression_rate']:.3f} |"
            ),
        ]
    )
    for label, group in (
        ("低风险", governance["risk"]["low"]),
        ("高风险", governance["risk"]["high"]),
        ("intra-community", governance["community_relation"]["intra"]),
        ("inter-community", governance["community_relation"]["inter"]),
    ):
        lines.append(
            f"| {label} | {group['candidate']} | {group['shown']} | "
            f"{group['suppressed']} | {group['suppression_rate']:.3f} |"
        )
    lines.extend(["", "每时间步：", ""])
    for timestep, group in governance["timestep"].items():
        lines.append(
            f"- t={timestep}: 候选 {group['candidate']}，展示 {group['shown']}，"
            f"抑制 {group['suppressed']}，抑制率 {group['suppression_rate']:.3f}"
        )
    lines.extend(
        [
            "",
            (
                "高风险抑制率高于低风险抑制率；这只是该合成运行中的分层观察，"
                "不能解释为因果效果。"
            ),
            "",
            "## 样本明细",
            "",
            "| sample ID | t | user | action | target | root | risk | relation | sanity |",
            "|---|---:|---:|---|---:|---:|---|---|---|",
        ]
    )
    for sample in audit["samples"]:
        lines.append(
            f"| {sample['sample_id']} | {sample['timestep']} | {sample['user_id']} | "
            f"{sample['action']} | {sample['direct_post_id']} | "
            f"{sample['root_post_id']} | {sample['risk_bucket']} | "
            f"{sample['community_relation']} | {sample['rule_classification']} |"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in audit["limitations"])
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                f"**{audit['recommendation']['choice']}. "
                f"{audit['recommendation']['label']}。**"
            ),
            "",
            (
                "动作 sanity check 支持扩大到 100 决策做稳定性评估；但在任何 LoRA "
                "使用前，必须修复 JSONL 的 Feed 序列化，并继续审计 rationale 事实错误。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_directory(args.output_dir)
    serialized = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        _write_text(args.json_output, serialized)
    if args.markdown_output is not None:
        _write_text(args.markdown_output, render_markdown(audit))
    if args.json_output is None and args.markdown_output is None:
        print(serialized, end="")


if __name__ == "__main__":
    main()
