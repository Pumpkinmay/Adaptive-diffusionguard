from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    DecisionSnapshot,
    build_response_format,
)
from adaptive_diffusionguard.storage.decision_snapshots import DecisionSnapshotStore
from scripts.validate_teacher_dataset import validate_dataset
from training.dataset_pipeline import (
    DatasetIntegrityError,
    build_from_decision_snapshots,
    reconstruct_legacy_dataset,
    serialize_jsonl,
    write_dataset,
)


def _snapshot(*, timestep: int = 0, content: str = "visible post") -> DecisionSnapshot:
    feed = {"success": True, "posts": [{"post_id": 1, "content": content}]}
    mask = ActionMask(("ignore", "repost:1", "quote:1", "report:1"))
    response_format = build_response_format(mask)
    messages = [
        {"role": "system", "content": "synthetic teacher"},
        {
            "role": "user",
            "content": json.dumps(
                {"visible_feed": feed["posts"], "legal_choice_ids": mask.choices}
            ),
        },
    ]
    return DecisionSnapshot.capture(
        user_id=0,
        timestep=timestep,
        feed=feed,
        visible_post_ids={1},
        mask=mask,
        response_format=response_format,
        messages=messages,
        state_identifier=f"state-{timestep}",
    )


def _native_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE trace (user_id INTEGER, created_at INTEGER, action TEXT, info TEXT)"
    )
    connection.execute(
        "INSERT INTO trace VALUES (0, 0, 'refresh', ?)",
        (json.dumps({"posts": [{"post_id": 1, "content": "visible post"}]}),),
    )
    connection.commit()
    return connection


def test_snapshot_migration_is_idempotent_and_native_build_is_deterministic(
    tmp_path: Path,
) -> None:
    database = tmp_path / "native.db"
    connection = _native_database(database)
    store = DecisionSnapshotStore(connection)
    pending = store.create_pending(
        _snapshot(),
        run_id="test-run",
        agent_id=0,
        decision_sequence=1,
        user_profile="synthetic profile",
        community="community-0",
        behavior_history=["trace-1:refresh"],
    )
    status = connection.execute(
        "SELECT status FROM diffusionguard_decision_snapshot WHERE decision_id = ?",
        (pending.decision_id,),
    ).fetchone()[0]
    assert status == "pending"

    action_rowid = connection.execute(
        "INSERT INTO trace VALUES (0, 0, 'repost', ?) RETURNING rowid",
        (json.dumps({"reposted_id": 1, "new_post_id": 2}),),
    ).fetchone()[0]
    connection.commit()
    store.mark_succeeded(
        pending.decision_id,
        selected_choice_id="repost:1",
        rationale="safe synthetic rationale",
        action_trace_rowid=action_rowid,
    )
    DecisionSnapshotStore(connection)  # second migration must preserve the row
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM diffusionguard_decision_snapshot"
        ).fetchone()[0]
        == 1
    )
    connection.close()

    first = build_from_decision_snapshots(database)
    second = build_from_decision_snapshots(database)
    assert serialize_jsonl(first.examples) == serialize_jsonl(second.examples)
    example = first.examples[0]
    assert json.loads(example.feed_post)["posts"][0]["post_id"] == 1
    assert example.feed_post != json.dumps({"reposted_id": 1, "new_post_id": 2})
    assert example.action_trace_rowid == action_rowid
    assert example.behavior_history == ["trace-1:refresh"]
    assert "selected_choice_id" not in example.prompt()
    assert "safe synthetic rationale" not in example.prompt()
    assert "rationale" not in example.to_dict()


def test_failed_snapshot_is_retained_but_excluded_from_degraded_export(
    tmp_path: Path,
) -> None:
    database = tmp_path / "failed.db"
    connection = _native_database(database)
    store = DecisionSnapshotStore(connection)
    pending = store.create_pending(
        _snapshot(),
        run_id="test-run",
        agent_id=0,
        decision_sequence=1,
        user_profile="synthetic profile",
        community="community-0",
        behavior_history=["trace-1:refresh"],
    )
    store.mark_failed(pending.decision_id, "json_validate_failed")
    connection.close()

    with pytest.raises(DatasetIntegrityError, match="failed decision snapshots"):
        build_from_decision_snapshots(database)
    degraded = build_from_decision_snapshots(database, allow_failed=True)
    assert degraded.examples == ()
    assert degraded.report["excluded_failed_decisions"] == 1
    assert degraded.report["status"] == "degraded"


def test_snapshot_store_rejects_credentials_and_invalid_trace_binding(
    tmp_path: Path,
) -> None:
    connection = _native_database(tmp_path / "secret.db")
    store = DecisionSnapshotStore(connection)
    with pytest.raises(ValueError, match="credential-like"):
        store.create_pending(
            _snapshot(content="gsk_" + "unit_test_secret_123456789"),
            run_id="test-run",
            agent_id=0,
            decision_sequence=1,
            user_profile="synthetic",
            community="community-0",
            behavior_history=["trace-1:refresh"],
        )
    pending = store.create_pending(
        _snapshot(),
        run_id="test-run",
        agent_id=0,
        decision_sequence=2,
        user_profile="synthetic",
        community="community-0",
        behavior_history=["trace-1:refresh"],
    )
    wrong_rowid = connection.execute(
        "INSERT INTO trace VALUES (0, 0, 'report_post', ?) RETURNING rowid",
        (json.dumps({"post_id": 1, "report_id": 1}),),
    ).fetchone()[0]
    connection.commit()
    with pytest.raises(ValueError, match="does not match"):
        store.mark_succeeded(
            pending.decision_id,
            selected_choice_id="repost:1",
            rationale="safe rationale",
            action_trace_rowid=wrong_rowid,
        )
    connection.close()


def _legacy_fixture(root: Path, *, ambiguous: bool = False) -> dict[str, Path]:
    root.mkdir()
    database_path = root / "real_llm_validation.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE trace (
            user_id INTEGER, created_at INTEGER, action TEXT, info TEXT
        );
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY, user_id INTEGER,
            original_post_id INTEGER, quote_content TEXT, content TEXT
        );
        CREATE TABLE diffusionguard_impression (
            impression_id INTEGER PRIMARY KEY, run_id TEXT, timestep INTEGER,
            user_id INTEGER, post_id INTEGER, root_post_id INTEGER,
            user_community TEXT, author_community TEXT, base_score REAL,
            risk_score REAL, omega_intra REAL, omega_inter REAL,
            keep_probability REAL, shown INTEGER, source TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO post VALUES (?, ?, ?, ?, ?)",
        [
            (1, 0, None, None, "Verified synthetic weather update."),
            (2, 1, 1, None, ""),
        ],
    )
    feed = json.dumps(
        {"success": True, "posts": [{"post_id": 1, "content": "verified"}]}
    )
    connection.execute("INSERT INTO trace VALUES (1, 0, 'refresh', ?)", (feed,))
    if ambiguous:
        connection.execute("INSERT INTO trace VALUES (1, 0, 'refresh', ?)", (feed,))
    trace_rowid = connection.execute(
        "INSERT INTO trace VALUES (1, 0, 'repost', ?) RETURNING rowid",
        (json.dumps({"reposted_id": 1, "new_post_id": 2}),),
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO diffusionguard_impression VALUES "
        "(1, 'legacy-run', 0, 1, 1, 1, 'community-1', 'community-0', "
        "1.0, 0.0, 0.4, 0.0, 1.0, 1, 'recommendation')"
    )
    connection.commit()
    connection.close()

    source_path = root / "groq_teacher_trajectories.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "sample_id": f"trace-{trace_rowid}",
                "user_profile": "synthetic user 1",
                "community": "community-1",
                "feed_post": json.dumps({"reposted_id": 1, "new_post_id": 2}),
                "neighbor_interactions": [],
                "behavior_history": [],
                "platform_notice": "",
                "label": {
                    "action": "repost",
                    "confidence": 1.0,
                    "reason": "The user reposted their own original post.",
                },
                "label_source": "teacher_synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quality_path = root / "teacher_quality_audit.json"
    quality_path.write_text(
        json.dumps(
            {
                "quality": {
                    "rationale_issue_sample_ids": {
                        "factual_author_mismatch": [f"trace-{trace_rowid}"]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary_path = root / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    profiles_path = root / "synthetic_profiles.csv"
    profiles_path.write_text("username\nsynthetic_agent_1\n", encoding="utf-8")
    return {
        "database": database_path,
        "profiles": profiles_path,
        "quality": quality_path,
        "source": source_path,
        "summary": summary_path,
    }


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def test_legacy_reconstruction_is_unique_deterministic_and_excludes_bad_rationale(
    tmp_path: Path,
) -> None:
    paths = _legacy_fixture(tmp_path / "legacy")
    original_hashes = _hashes(paths)

    first = reconstruct_legacy_dataset(
        paths["database"], paths["source"], quality_audit_path=paths["quality"]
    )
    second = reconstruct_legacy_dataset(
        paths["database"], paths["source"], quality_audit_path=paths["quality"]
    )

    assert serialize_jsonl(first.examples) == serialize_jsonl(second.examples)
    assert first.report == second.report
    assert first.report["status"] == "success"
    assert first.report["unique_refresh_matches"] == 1
    example = first.examples[0]
    assert example.action_label_eligible is True
    assert example.rationale_training_eligible is False
    assert example.rationale_quality_status == "factual_relation_error"
    assert example.label.action == "repost"
    assert example.label.reason == ""
    assert json.loads(example.feed_post)["posts"][0]["post_id"] == 1
    assert example.feed_post != json.dumps(
        {"reposted_id": 1, "new_post_id": 2}, sort_keys=True
    )
    assert (
        first.report["rationale_quality_exclusions"][0]["original_rationale"]
        == "The user reposted their own original post."
    )
    assert _hashes(paths) == original_hashes

    output = tmp_path / "corrected.jsonl"
    write_dataset(first.examples, output)
    validation = validate_dataset(output, database=paths["database"])
    assert validation["status"] == "success"
    assert validation["action_label_eligible"] == 1
    assert validation["rationale_training_eligible"] == 0


def test_legacy_ambiguous_refresh_is_unavailable_and_not_exported(
    tmp_path: Path,
) -> None:
    paths = _legacy_fixture(tmp_path / "ambiguous", ambiguous=True)

    build = reconstruct_legacy_dataset(
        paths["database"], paths["source"], quality_audit_path=paths["quality"]
    )

    assert build.report["status"] == "failed"
    assert build.report["matched_decisions"] == 0
    assert build.examples == ()
    assert any(
        row["reason"] == "legacy_refresh_match_is_not_unique"
        for row in build.report["ambiguous_or_unavailable"]
    )


def test_legacy_source_credentials_are_rejected_before_output(tmp_path: Path) -> None:
    paths = _legacy_fixture(tmp_path / "credential")
    source = paths["source"].read_text(encoding="utf-8")
    paths["source"].write_text(
        source.replace(
            "The user reposted their own original post.",
            "gsk_" + "unit_test_secret_123456789",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetIntegrityError, match="credential-like"):
        reconstruct_legacy_dataset(
            paths["database"],
            paths["source"],
            quality_audit_path=paths["quality"],
        )


def test_validator_rejects_current_or_future_history(tmp_path: Path) -> None:
    paths = _legacy_fixture(tmp_path / "history")
    build = reconstruct_legacy_dataset(
        paths["database"], paths["source"], quality_audit_path=paths["quality"]
    )
    row = build.examples[0].to_dict()
    row["behavior_history"] = [f"trace-{row['action_trace_rowid']}:repost"]
    output = tmp_path / "leaking.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validate_dataset(output, database=paths["database"])

    assert report["status"] == "failed"
    assert any("current or future" in error for error in report["errors"])
