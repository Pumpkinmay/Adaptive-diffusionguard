from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from scripts.audit_teacher_quality import audit_directory, render_markdown


def _write_fixture(root: Path, *, omit_target_impression: bool = False) -> None:
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "secret_scan_passed": True,
                "teacher_examples": 4,
                "timesteps": 2,
                "validation_mode": "strict_structured_action_mask",
            }
        ),
        encoding="utf-8",
    )
    with (root / "synthetic_profiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["username", "description", "user_char"]
        )
        writer.writeheader()
        for user_id in range(3):
            writer.writerow(
                {
                    "username": f"synthetic_agent_{user_id}",
                    "description": "synthetic",
                    "user_char": "careful",
                }
            )

    database = sqlite3.connect(root / "real_llm_validation.db")
    database.executescript(
        """
        CREATE TABLE user (user_id INTEGER PRIMARY KEY);
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            original_post_id INTEGER,
            content TEXT,
            quote_content TEXT,
            created_at INTEGER
        );
        CREATE TABLE report (
            report_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            report_reason TEXT,
            created_at INTEGER
        );
        CREATE TABLE trace (
            user_id INTEGER,
            created_at INTEGER,
            action TEXT,
            info TEXT
        );
        CREATE TABLE diffusionguard_impression (
            impression_id INTEGER PRIMARY KEY,
            run_id TEXT,
            timestep INTEGER,
            user_id INTEGER,
            post_id INTEGER,
            root_post_id INTEGER,
            user_community TEXT,
            author_community TEXT,
            base_score REAL,
            risk_score REAL,
            omega_intra REAL,
            omega_inter REAL,
            keep_probability REAL,
            shown INTEGER,
            source TEXT
        );
        """
    )
    database.executemany("INSERT INTO user VALUES (?)", [(0,), (1,), (2,)])
    database.executemany(
        "INSERT INTO post VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 0, None, "Verified synthetic weather update.", None, 0),
            (2, 1, None, "Unverified synthetic health claim.", None, 0),
            (3, 2, 2, "", None, 0),
            (4, 0, 1, "", None, 1),
        ],
    )
    traces = [
        (0, 0, "refresh", {"posts": [{"post_id": 3, "content": "risk repost"}]}),
        (0, 0, "report_post", {"post_id": 3, "report_id": 1}),
        (1, 0, "refresh", {"posts": [{"post_id": 1, "content": "low"}]}),
        (1, 0, "do_nothing", {}),
        (2, 1, "refresh", {"posts": [{"post_id": 2, "content": "high"}]}),
        (2, 1, "do_nothing", {}),
        (0, 1, "refresh", {"posts": [{"post_id": 1, "content": "low"}]}),
        (0, 1, "repost", {"reposted_id": 1, "new_post_id": 4}),
    ]
    database.executemany(
        "INSERT INTO trace VALUES (?, ?, ?, ?)",
        [(user, step, action, json.dumps(info)) for user, step, action, info in traces],
    )
    database.execute(
        "INSERT INTO report VALUES (?, ?, ?, ?, ?)",
        (1, 0, 3, "unverified health claim", 0),
    )
    impressions = [
        (1, 0, 0, 3, 2, "community-0", "community-1", 1.0),
        (2, 0, 1, 1, 1, "community-1", "community-0", 0.0),
        (3, 1, 2, 2, 2, "community-0", "community-1", 1.0),
        (4, 1, 0, 1, 1, "community-0", "community-0", 0.0),
    ]
    if omit_target_impression:
        impressions = impressions[1:]
    database.executemany(
        """
        INSERT INTO diffusionguard_impression VALUES
        (?, 'fixture', ?, ?, ?, ?, ?, ?, 1.0, ?, 0.4, 0.0, 1.0, 1, 'recommendation')
        """,
        impressions,
    )
    database.commit()
    database.close()

    teacher_rows = [
        ("trace-2", "report", "The unverified health claim should be reported."),
        ("trace-4", "ignore", "The verified weather update needs no action."),
        ("trace-6", "ignore", "I will ignore this unverified health claim."),
        ("trace-8", "repost", "The verified weather update is safe to repost."),
    ]
    with (root / "groq_teacher_trajectories.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for sample_id, action, reason in teacher_rows:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "user_profile": "synthetic",
                        "community": "community-0",
                        "feed_post": "{}",
                        "neighbor_interactions": [],
                        "behavior_history": [],
                        "platform_notice": "",
                        "label": {
                            "action": action,
                            "confidence": 1.0,
                            "reason": reason,
                        },
                        "label_source": "teacher_synthetic",
                    }
                )
                + "\n"
            )


def test_root_mapping_and_risk_cross_table(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_fixture(output)

    audit = audit_directory(output)
    sample = next(row for row in audit["samples"] if row["sample_id"] == "trace-2")

    assert sample["direct_post_id"] == 3
    assert sample["root_post_id"] == 2
    assert sample["root_content"] == "Unverified synthetic health claim."
    assert sample["risk_score"] == 1.0
    assert audit["quality"]["report_high_risk_count"] == 1
    assert audit["quality"]["repost_low_risk_count"] == 1
    assert audit["quality"]["rule_aligned_numerator"] == 2
    assert audit["quality"]["rule_aligned_denominator"] == 2


def test_ignore_feed_judgment_is_separate_from_rule_aligned_rate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_fixture(output)

    quality = audit_directory(output)["quality"]

    assert quality["ignore_feed_only_low_risk_count"] == 1
    assert quality["ignore_feed_contains_high_risk_count"] == 1
    assert quality["missed_intervention_candidate_count"] == 1
    assert quality["manual_review_sample_ids"] == ["trace-6"]
    assert quality["rule_aligned_denominator"] == 2


def test_missing_impression_is_null_with_reason(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_fixture(output, omit_target_impression=True)

    audit = audit_directory(output)
    sample = next(row for row in audit["samples"] if row["sample_id"] == "trace-2")

    assert sample["target_impression_id"] is None
    assert sample["risk_score"] == 1.0  # available from another root-2 impression
    assert sample["null_reasons"]["target_impression_id"].startswith(
        "target action does not map"
    )
    assert "target action does not map" in " ".join(sample["link_errors"])
    assert (
        audit["integrity"]["all_samples_linked_to_user_timestep_feed_and_impressions"]
        is False
    )


def test_audit_and_markdown_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_fixture(output)

    first = audit_directory(output)
    second = audit_directory(output)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert render_markdown(first) == render_markdown(second)
    assert first["integrity"]["credential_pattern_scan"]["env_scanned"] is False


def test_teacher_feed_field_is_not_mistaken_for_refresh_feed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_fixture(output)

    integrity = audit_directory(output)["integrity"]

    assert integrity["teacher_feed_post_mismatch_count"] == 4
    assert integrity["jsonl_valid_line_count"] == 4
    assert integrity["sample_ids_one_to_one_with_successful_traces"] is True
