"""Offline, fail-closed validation for reconstructed teacher JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from training.dataset_pipeline import ACTION_MAP, ACTION_RESULT_KEYS, SNAPSHOT_TABLE
from training.schemas import BehaviorExample

SECRET_PATTERN = re.compile(r"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")
VALID_PROVENANCE = frozenset(
    {"decision_snapshot", "legacy_refresh_trace_reconstruction"}
)


def _target(choice_id: str) -> int | None:
    action, separator, value = choice_id.partition(":")
    if action == "ignore":
        return None
    if not separator:
        raise ValueError("targeted selected_choice_id has no post_id")
    return int(value)


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("row is not an object")
            rows.append(row)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"line {line_number}: invalid JSON object: {exc}")
    if not rows:
        errors.append("dataset contains no samples")
    return rows, errors


def _validate_database_bindings(
    rows: list[dict[str, Any]], database: Path, errors: list[str]
) -> None:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        has_snapshots = (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (SNAPSHOT_TABLE,),
            ).fetchone()
            is not None
        )
        dataset_trace_ids: set[int] = set()
        for row in rows:
            sample_id = str(row.get("sample_id", "<missing>"))
            trace_rowid = row.get("action_trace_rowid")
            if not isinstance(trace_rowid, int):
                errors.append(f"{sample_id}: action_trace_rowid is not an integer")
                continue
            if trace_rowid in dataset_trace_ids:
                errors.append(f"{sample_id}: duplicate action_trace_rowid")
            dataset_trace_ids.add(trace_rowid)
            trace = connection.execute(
                "SELECT action FROM trace WHERE rowid = ?", (trace_rowid,)
            ).fetchone()
            if trace is None or trace[0] not in ACTION_MAP:
                errors.append(f"{sample_id}: bound successful action trace is missing")
                continue
            label_action = row.get("label", {}).get("action")
            if ACTION_MAP[trace[0]] != label_action:
                errors.append(f"{sample_id}: action label disagrees with trace")
        if has_snapshots:
            succeeded = {
                (str(decision_id), int(trace_rowid))
                for decision_id, trace_rowid in connection.execute(
                    f"SELECT decision_id, action_trace_rowid FROM {SNAPSHOT_TABLE} "
                    "WHERE status='succeeded'"
                )
            }
            native = {
                (str(row.get("sample_id")), int(row["action_trace_rowid"]))
                for row in rows
                if row.get("provenance") == "decision_snapshot"
                and isinstance(row.get("action_trace_rowid"), int)
            }
            if native != succeeded:
                errors.append(
                    "native dataset is not one-to-one with succeeded snapshots"
                )
    finally:
        connection.close()


def validate_dataset(path: Path, *, database: Path | None = None) -> dict[str, Any]:
    rows, errors = _load_rows(path)
    examples: list[BehaviorExample] = []
    seen_ids: set[str] = set()
    sequences: list[tuple[int, str]] = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id", f"line-{index}"))
        try:
            example = BehaviorExample.from_dict(row)
        except Exception as exc:  # noqa: BLE001 - validation boundary
            errors.append(f"{sample_id}: invalid BehaviorExample: {exc}")
            continue
        examples.append(example)
        if example.sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate decision_id")
        seen_ids.add(example.sample_id)
        if example.decision_sequence is None:
            errors.append(f"{sample_id}: decision_sequence is missing")
        else:
            sequences.append((example.decision_sequence, example.sample_id))
        for required in (
            "action_label_eligible",
            "rationale_training_eligible",
            "provenance",
            "selected_choice_id",
        ):
            if required not in row:
                errors.append(
                    f"{sample_id}: required audit field {required} is missing"
                )
        if example.provenance not in VALID_PROVENANCE:
            errors.append(f"{sample_id}: provenance is not explicit")
        try:
            feed = json.loads(example.feed_post)
            if not isinstance(feed, dict):
                raise TypeError("Feed is not an object")
            posts = feed.get("posts")
            if not isinstance(posts, list):
                raise TypeError("Feed posts is not an array")
            feed_ids = [int(post["post_id"]) for post in posts]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{sample_id}: invalid normalized Feed: {exc}")
            continue
        if sorted(feed_ids) != sorted(example.visible_post_ids):
            errors.append(f"{sample_id}: visible_post_ids disagree with Feed")
        if not feed_ids and not example.feed_empty_reason:
            errors.append(f"{sample_id}: empty Feed has no explicit reason")
        if ACTION_RESULT_KEYS.intersection(feed):
            errors.append(f"{sample_id}: Feed contains current action-result keys")
        if example.selected_choice_id not in example.legal_choice_ids:
            errors.append(f"{sample_id}: selected choice is not legal")
        else:
            try:
                target = _target(str(example.selected_choice_id))
                if target is not None and target not in feed_ids:
                    errors.append(f"{sample_id}: selected target is not visible")
                selected_action = str(example.selected_choice_id).partition(":")[0]
                if selected_action != example.label.action:
                    errors.append(
                        f"{sample_id}: selected choice disagrees with action label"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(f"{sample_id}: invalid selected choice: {exc}")
        if example.action_trace_rowid is None:
            errors.append(f"{sample_id}: action_trace_rowid is missing")
        else:
            for token in example.behavior_history:
                match = re.fullmatch(r"trace-(\d+):([a-z_]+)", token)
                if match is None:
                    errors.append(f"{sample_id}: invalid history token {token}")
                elif int(match.group(1)) >= example.action_trace_rowid:
                    errors.append(
                        f"{sample_id}: history contains current or future action"
                    )
        try:
            prompt_payload = json.loads(example.prompt().split("\n", 1)[1])
        except (IndexError, json.JSONDecodeError) as exc:
            errors.append(f"{sample_id}: prompt payload is invalid: {exc}")
        else:
            if "selected_choice_id" in prompt_payload or "label" in prompt_payload:
                errors.append(f"{sample_id}: label audit fields leaked into input")
            allowed = {
                "behavior_history",
                "community",
                "feed_post",
                "legal_choice_ids",
                "neighbor_interactions",
                "platform_notice",
                "user_profile",
            }
            if set(prompt_payload) != allowed:
                errors.append(f"{sample_id}: prompt input fields are not canonical")
        if not example.rationale_training_eligible and example.label.reason:
            errors.append(
                f"{sample_id}: ineligible rationale text remains in training label"
            )
        if SECRET_PATTERN.search(json.dumps(row, ensure_ascii=False)):
            errors.append(f"{sample_id}: credential-like pattern found")

    expected_order = sorted(sequences)
    actual_order = [
        (example.decision_sequence, example.sample_id)
        for example in examples
        if example.decision_sequence is not None
    ]
    if actual_order != expected_order:
        errors.append("samples are not in deterministic decision order")
    if database is not None:
        _validate_database_bindings(rows, database, errors)
    report = {
        "action_label_eligible": sum(
            bool(row.get("action_label_eligible")) for row in rows
        ),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decision_id_unique": len(seen_ids) == len(rows),
        "errors": sorted(set(errors)),
        "provenance_values": sorted({str(row.get("provenance")) for row in rows}),
        "rationale_training_eligible": sum(
            bool(row.get("rationale_training_eligible")) for row in rows
        ),
        "sample_count": len(rows),
        "status": "success" if not errors else "failed",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset, database=args.database)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
