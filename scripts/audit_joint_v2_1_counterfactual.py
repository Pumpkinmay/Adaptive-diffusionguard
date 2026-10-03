"""Read-only counterfactual v2.1 audit of the frozen real v2 micro-pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

KNOWN_DECISION_ID = (
    "joint-v2-strong-community-static_cosref-61001:"
    "t000001:u000031:d000002"
)
SECRET_PATTERN = re.compile(rb"(?i)(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{12,}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _tree_digest(output: Path) -> str:
    digest = hashlib.sha256()
    excluded = "v2_1_counterfactual_consistency_audit.json"
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == excluded:
            continue
        digest.update(str(path.relative_to(output)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def audit(output: Path) -> dict[str, Any]:
    before = _tree_digest(output)
    decisions = _read_jsonl(output / "per_decision.jsonl")
    by_id = {str(row["decision_id"]): row for row in decisions}
    records: list[dict[str, Any]] = []
    selected_mapping_mismatches: list[str] = []
    known_verified = False
    for database in sorted(output.glob("units/*/*/experiment.db")):
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT decision_id, legal_choice_ids_json, rationale, "
                "selected_choice_id, status FROM diffusionguard_decision_snapshot "
                "ORDER BY decision_id"
            ).fetchall()
            for snapshot in rows:
                decision_id = str(snapshot["decision_id"])
                row = by_id[decision_id]
                legal = tuple(json.loads(snapshot["legal_choice_ids_json"]))
                index = int(row["choice_index"])
                indexed_choice = legal[index] if 0 <= index < len(legal) else None
                selected = str(snapshot["selected_choice_id"] or "")
                mapping_consistent = indexed_choice == selected
                if not mapping_consistent:
                    selected_mapping_mismatches.append(decision_id)
                record: dict[str, Any] = {
                    "decision_id": decision_id,
                    "v2_choice_index": index,
                    "v2_indexed_choice_id": indexed_choice,
                    "v2_selected_choice_id": selected,
                    "v2_index_to_execution_consistent": mapping_consistent,
                    "v2_1_action_type": None,
                    "v2_1_target_post_id": None,
                    "v2_1_quote_text": None,
                    "v2_1_direct_validation_status": "not_reconstructable",
                    "not_reconstructable_reason": (
                        "v2 response did not contain action_type, target_post_id, "
                        "or quote_text"
                    ),
                }
                if decision_id == KNOWN_DECISION_ID:
                    rationale = str(snapshot["rationale"] or "")
                    known_verified = all(
                        (
                            selected == "report:2",
                            row.get("selected_post_id") == 2,
                            row.get("root_risk_score") == 0.0,
                            "post 3" in rationale.lower(),
                            "report" in rationale.lower(),
                        )
                    )
                    record["known_record_verified"] = known_verified
                    record["counterfactual_explicit_semantic_fields"] = {
                        "action_type": "report",
                        "target_post_id": 3,
                    }
                    record["counterfactual_v2_1_result"] = (
                        "rejected_choice_target_mismatch"
                        if known_verified
                        else "not_reconstructable"
                    )
                    record["counterfactual_basis"] = (
                        "The preserved rationale explicitly says report post 3; "
                        "the indexed and executed option is report post 2. If the "
                        "new explicit fields expressed that stated intent, v2.1 "
                        "would reject target_post_id=3 against report:2."
                    )
                records.append(record)
    if set(by_id) != {str(row["decision_id"]) for row in records}:
        raise RuntimeError("snapshot/decision population mismatch")
    result = {
        "audit_type": "v2_to_v2_1_counterfactual_semantic_consistency",
        "source_output": str(output),
        "source_protocol": "fixed-choice-index-v2",
        "target_protocol": "fixed-choice-semantic-v2.1",
        "decision_count": len(records),
        "v2_index_to_execution_consistent_count": sum(
            bool(row["v2_index_to_execution_consistent"]) for row in records
        ),
        "v2_index_to_execution_mismatch_decision_ids": sorted(
            selected_mapping_mismatches
        ),
        "v2_1_directly_reconstructable_count": 0,
        "v2_1_not_reconstructable_count": len(records),
        "known_decision_id": KNOWN_DECISION_ID,
        "known_record_verified": known_verified,
        "known_record_counterfactual_rejected": known_verified,
        "interpretation_limit": (
            "The old v2 response lacks the new redundant semantic fields. This "
            "audit proves the preserved known record would be rejected only under "
            "the explicit counterfactual fields supported by its unambiguous "
            "rationale; it does not claim v2.1 has changed real model behavior."
        ),
        "records": sorted(records, key=lambda row: row["decision_id"]),
        "source_tree_sha256_before": before,
        "remote_api_calls": 0,
        "teacher_or_training_samples_created": 0,
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True).encode()
    if SECRET_PATTERN.search(encoded):
        raise RuntimeError("credential-like pattern detected in audit")
    result["secret_scan_passed"] = True
    after = _tree_digest(output)
    result["source_tree_sha256_after"] = after
    result["source_tree_hash_unchanged"] = before == after
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cosref-llm-joint-v2-groq-micro"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.output)
    _write_json(
        args.output / "v2_1_counterfactual_consistency_audit.json",
        result,
    )
    print(
        json.dumps(
            {
                "decision_count": result["decision_count"],
                "known_record_counterfactual_rejected": result[
                    "known_record_counterfactual_rejected"
                ],
                "source_tree_hash_unchanged": result[
                    "source_tree_hash_unchanged"
                ],
                "v2_1_not_reconstructable_count": result[
                    "v2_1_not_reconstructable_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
