"""Plot repository evaluation results from JSON summaries.

This script deliberately refuses to substitute aggregate repost counts for a
risk-adoption metric.  The strategy plot is generated only when every baseline
summary contains an explicit risk-adoption/final-adoption field.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.switch_backend("Agg")


BACKENDS = ("rule-based", "random", "Groq-LLM")
BACKEND_COLORS = ("#2E8B57", "#C73E3A", "#377EB8")
QUALITY_METRICS = (
    ("aligned_rate", "Aligned", ""),
    ("acceptable_rate", "Acceptable", "//"),
    ("concerning_rate", "Concerning", "xx"),
)
ADVERSE_COUNTS = (
    ("repost_high_risk_count", "High-risk reposts", ""),
    ("report_low_risk_count", "Low-risk reports", "//"),
)
BASELINES = (
    "no_intervention",
    "global_throttle",
    "static_cosref",
    "dynamic_cosref",
)
def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return data


def _groq_quality_path(runs_dir: Path) -> Path | None:
    root = runs_dir / "teacher-eval-100-groq"
    candidates = (
        root / "batches" / "batch-01" / "quality_summary.json",
        root / "quality_summary.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.rglob("*quality*.json")) if root.exists() else []
    return matches[0] if matches else None


def load_teacher_quality(runs_dir: Path) -> list[dict[str, Any] | None]:
    paths: list[Path | None] = [
        runs_dir / "teacher-eval-100-fake" / "quality_summary.json",
        runs_dir / "teacher-eval-100-random" / "quality_summary.json",
        _groq_quality_path(runs_dir),
    ]
    summaries: list[dict[str, Any] | None] = []
    for backend, path in zip(BACKENDS, paths):
        if path is None or not path.is_file():
            warnings.warn(
                f"{backend} quality summary was not found; plotting N/A.",
                stacklevel=2,
            )
            summaries.append(None)
        else:
            summaries.append(_read_json(path))
    return summaries


def plot_teacher_quality(runs_dir: Path, output: Path) -> None:
    summaries = load_teacher_quality(runs_dir)
    figure, (quality_ax, adverse_ax) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.8),
        gridspec_kw={"width_ratios": (1.75, 1.0)},
    )
    positions = list(range(len(BACKENDS)))
    width = 0.22

    for metric_index, (key, label, hatch) in enumerate(QUALITY_METRICS):
        offset = (metric_index - 1) * width
        for backend_index, summary in enumerate(summaries):
            value = math.nan if summary is None else float(summary[key]) * 100.0
            bar = quality_ax.bar(
                backend_index + offset,
                value,
                width,
                color=BACKEND_COLORS[backend_index],
                edgecolor="white",
                hatch=hatch,
                linewidth=0.8,
                alpha=0.9,
            )[0]
            if math.isnan(value):
                quality_ax.text(
                    backend_index + offset,
                    2.0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                )
            elif key == "concerning_rate":
                count = int(summary["concerning_count"])
                quality_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.6,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    quality_ax.set_ylabel("Rate among scored decisions (%)")
    quality_ax.set_xticks(positions, BACKENDS)
    quality_ax.set_ylim(0, 108)
    quality_ax.set_title("Synthetic rubric rates", y=1.16, fontweight="bold")
    quality_ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    quality_ax.spines[["top", "right"]].set_visible(False)
    quality_ax.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="white", hatch=hatch, label=label)
            for _, label, hatch in QUALITY_METRICS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=3,
        frameon=False,
    )

    adverse_width = 0.28
    for metric_index, (key, label, hatch) in enumerate(ADVERSE_COUNTS):
        offset = (metric_index - 0.5) * adverse_width
        for backend_index, summary in enumerate(summaries):
            value = math.nan if summary is None else int(summary[key])
            adverse_ax.bar(
                backend_index + offset,
                value,
                adverse_width,
                color=BACKEND_COLORS[backend_index],
                edgecolor="white",
                hatch=hatch,
                linewidth=0.8,
                alpha=0.9,
            )
            label_text = "N/A" if math.isnan(value) else str(value)
            label_y = 0.15 if math.isnan(value) else value + 0.25
            adverse_ax.text(
                backend_index + offset,
                label_y,
                label_text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    adverse_ax.set_ylabel("Decision count")
    adverse_ax.set_xticks(positions, BACKENDS)
    adverse_ax.set_title(
        "Safety-adverse action counts", y=1.16, fontweight="bold"
    )
    adverse_ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    adverse_ax.spines[["top", "right"]].set_visible(False)
    adverse_ax.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="white", hatch=hatch, label=label)
            for _, label, hatch in ADVERSE_COUNTS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=2,
        frameon=False,
    )

    figure.suptitle(
        "100-decision teacher evaluation: rule-based vs random vs Groq LLM",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        "Rule-based and random: 100 decisions each; Groq: batch-01 only "
        "(20 decisions). Different sample sizes are not directly comparable.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.13,
        top=0.72,
        wspace=0.28,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def load_strategy_summaries(summary_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for baseline in BASELINES:
        path = summary_dir / f"{baseline}-summary.json"
        data = _read_json(path)
        summaries.append(
            {
                "baseline": baseline,
                "intervention_cost": float(data["intervention_cost"]),
                "successful_reposts": int(data["successful_reposts"]),
            }
        )
    return summaries


def plot_strategy_tradeoff(summary_dir: Path, output: Path) -> None:
    summaries = load_strategy_summaries(summary_dir)
    colors = ("#555555", "#E69F00", "#56B4E9", "#CC79A7")
    markers = ("o", "s", "^", "D")
    label_offsets = ((8, 8), (-110, 12), (-105, 12), (8, 12))
    figure, axis = plt.subplots(figsize=(10.5, 6.8))
    for summary, color, marker, offset in zip(
        summaries, colors, markers, label_offsets
    ):
        x_value = summary["intervention_cost"]
        y_value = summary["successful_reposts"]
        axis.scatter(
            x_value,
            y_value,
            s=110,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.annotate(
            summary["baseline"],
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
        )
    axis.set_xlabel("Intervention cost (L1)")
    axis.set_ylabel("Successful reposts (diffusion-volume proxy)")
    axis.grid(alpha=0.25, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Exposure strategy trade-off (80-node SBM, 3 timesteps, scripted agents)",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.92,
        "synthetic, no LLM, descriptive smoke only",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--strategy-summary-dir",
        type=Path,
        default=Path("runs/all-baselines"),
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--strategy-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.teacher_only and args.strategy_only:
        raise ValueError("--teacher-only and --strategy-only are mutually exclusive")
    generated: list[Path] = []
    if not args.strategy_only:
        teacher_output = args.assets_dir / "teacher_eval_comparison.png"
        plot_teacher_quality(args.runs_dir, teacher_output)
        generated.append(teacher_output)
    if not args.teacher_only:
        strategy_output = args.assets_dir / "strategy_tradeoff.png"
        plot_strategy_tradeoff(args.strategy_summary_dir, strategy_output)
        generated.append(strategy_output)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
