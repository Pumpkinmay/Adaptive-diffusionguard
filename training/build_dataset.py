"""Build validated JSONL behavior data from OASIS traces or a teacher model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from adaptive_diffusionguard.llm.model_factory import LLMSettings, create_llm_model
from training.dataset_pipeline import (
    DatasetIntegrityError,
    build_from_decision_snapshots,
    has_decision_snapshots,
    reconstruct_legacy_dataset,
    serialize_jsonl,
    write_report,
)
from training.schemas import ActionOutput, BehaviorExample
from training.teacher import TeacherTrajectoryGenerator


def _load_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def load_jsonl(path: Path) -> list[BehaviorExample]:
    examples: list[BehaviorExample] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            examples.append(BehaviorExample.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid example on line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError("input JSONL is empty")
    return examples


def examples_from_oasis(
    db_path: Path,
    profiles: dict[str, Any],
    communities: dict[str, Any],
    label_source: str,
) -> Iterable[BehaviorExample]:
    """Export native snapshots; never infer Feed from an action trace row."""
    del profiles, communities
    if label_source != "teacher_synthetic":
        raise DatasetIntegrityError(
            "native decision snapshots contain distilled teacher labels"
        )
    if not has_decision_snapshots(db_path):
        raise DatasetIntegrityError(
            "database lacks native decision snapshots; use the explicit legacy "
            "refresh reconstruction path"
        )
    yield from build_from_decision_snapshots(db_path, allow_failed=True).examples


def demo_examples(label_source: str) -> list[BehaviorExample]:
    actions = ("repost", "quote", "report", "ignore")
    return [
        BehaviorExample(
            sample_id=f"demo-{index}",
            user_profile=f"synthetic profile {index}",
            community=f"community-{index % 4}",
            feed_post=f"synthetic post {index}",
            neighbor_interactions=["neighbor reposted" if index % 2 else ""],
            behavior_history=["ignore"],
            platform_notice="unverified claim" if action == "report" else "",
            label=ActionOutput(action=action, confidence=0.75, reason="demo label"),  # type: ignore[arg-type]
            label_source=label_source,  # type: ignore[arg-type]
            provenance="demo",
        )
        for index, action in enumerate(actions)
    ]


def write_jsonl(examples: Iterable[BehaviorExample], output: Path) -> int:
    materialized = list(examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_jsonl(materialized), encoding="utf-8")
    return len(materialized)


def build_teacher_dataset(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    settings = LLMSettings.from_env()
    model = create_llm_model(settings)
    if model is None:
        raise RuntimeError("teacher generation requires DIFFUSIONGUARD_ENABLE_LLM=true")
    generator = TeacherTrajectoryGenerator(model)
    generated = generator.generate(load_jsonl(args.teacher_input))
    count = write_jsonl(generated, args.output)
    metrics: dict[str, Any] = {
        **generator.metrics.to_dict(),
        "provider": settings.provider,
        "model": settings.llm_model,
        "label_source": "teacher_synthetic",
        "runtime": model.runtime.stats.to_dict(),
    }
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return count, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oasis-db", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--communities", type=Path)
    parser.add_argument("--teacher-input", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--legacy-teacher-source", type=Path)
    parser.add_argument("--quality-audit", type=Path)
    parser.add_argument("--legacy-refresh-reconstruction", action="store_true")
    parser.add_argument(
        "--label-source",
        choices=("observed", "teacher_synthetic"),
        default="observed",
        help="Teacher labels are behavior distillation, not human behavior.",
    )
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = sum(
        bool(value) for value in (args.demo, args.oasis_db, args.teacher_input)
    )
    if selected != 1:
        raise SystemExit("choose exactly one of --demo, --oasis-db, --teacher-input")
    if args.teacher_input:
        count, metrics = build_teacher_dataset(args)
        print(
            f"wrote {count} teacher_synthetic examples to {args.output}; "
            f"json_valid_rate={metrics['json_valid_rate']:.3f}; "
            f"action_valid_rate={metrics['action_valid_rate']:.3f}"
        )
        return
    if args.demo:
        examples = demo_examples(args.label_source)
    else:
        if args.legacy_refresh_reconstruction:
            if args.legacy_teacher_source is None or args.report_output is None:
                raise SystemExit(
                    "legacy reconstruction requires --legacy-teacher-source and "
                    "--report-output"
                )
            build = reconstruct_legacy_dataset(
                args.oasis_db,
                args.legacy_teacher_source,
                quality_audit_path=args.quality_audit,
                label_source=args.label_source,
            )
            write_report(build.report, args.report_output)
            if build.report["status"] != "success":
                raise SystemExit(2)
            examples = build.examples
        else:
            build = build_from_decision_snapshots(args.oasis_db)
            examples = build.examples
            if args.report_output is not None:
                write_report(build.report, args.report_output)
    count = write_jsonl(examples, args.output)
    print(f"wrote {count} validated examples to {args.output}")


if __name__ == "__main__":
    main()
