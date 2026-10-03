"""Preregistered COSREF exposure-mapping robustness pilot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .allocation import allocate_cosref_control, project_l1_cost
from .experiment import BASELINES, generate_directed_sbm, run_scenario
from .mixing import compute_mixing_statistics

CALIBRATION_METRICS = (
    "high_risk_exposures",
    "successful_risk_reposts",
    "risk_cascade_size",
    "intra_high_risk_exposures",
    "inter_high_risk_exposures",
    "community_coverage",
    "benign_exposure_loss",
    "realized_intervention_cost",
    "suppressed_impressions",
)
COMPARISON_METRICS = (
    "risk_cascade_size",
    "successful_risk_reposts",
    "high_risk_exposures",
    "intra_high_risk_exposures",
    "inter_high_risk_exposures",
    "community_coverage",
    "benign_exposure_loss",
    "nominal_project_cost_initial",
    "parameter_l1_cost_final",
    "realized_intervention_cost",
    "realized_intervention_cost_per_candidate",
    "suppressed_impressions",
)
COMPARATORS = (
    "no_intervention",
    "static_cosref",
    "dynamic_cosref",
    "global_throttle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write(path, text)


def bootstrap_mean_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a sample mean."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    distribution = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0

    def percentile(probability: float) -> float:
        position = probability * (len(distribution) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return distribution[lower]
        weight = position - lower
        return distribution[lower] * (1.0 - weight) + distribution[upper] * weight

    return percentile(tail), percentile(1.0 - tail)


def describe_values(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    low, high = bootstrap_mean_interval(
        values, resamples=resamples, seed=seed, confidence=confidence
    )
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_ci": [low, high],
        "confidence_level": confidence,
    }


def paired_differences(
    theory_rows: list[Mapping[str, object]],
    comparator_rows: list[Mapping[str, object]],
    metric: str,
) -> list[dict[str, object]]:
    """Match exactly by seed and compute theory minus comparator."""

    theory_by_seed = {int(row["seed"]): row for row in theory_rows}
    comparator_by_seed = {int(row["seed"]): row for row in comparator_rows}
    if len(theory_by_seed) != len(theory_rows) or len(comparator_by_seed) != len(
        comparator_rows
    ):
        raise ValueError("duplicate seed in paired comparison")
    if set(theory_by_seed) != set(comparator_by_seed):
        raise ValueError("paired comparison seed sets do not match")
    return [
        {
            "seed": seed,
            "theory": float(theory_by_seed[seed][metric]),
            "comparator": float(comparator_by_seed[seed][metric]),
            "difference": float(theory_by_seed[seed][metric])
            - float(comparator_by_seed[seed][metric]),
        }
        for seed in sorted(theory_by_seed)
    ]


def _choice_direction(omega: tuple[float, float]) -> str:
    if omega[0] < omega[1]:
        return "intra"
    if omega[1] < omega[0]:
        return "inter"
    return "balanced"


def _mean(rows: list[Mapping[str, object]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _select_candidate(
    rows: list[Mapping[str, object]],
    eligible: set[tuple[float, float]],
) -> tuple[float, float]:
    grouped: dict[tuple[float, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        pair = (float(row["omega_intra_initial"]), float(row["omega_inter_initial"]))
        if pair in eligible:
            grouped[pair].append(row)
    if set(grouped) != eligible:
        missing = sorted(eligible - set(grouped))
        raise ValueError(f"missing calibration observations for candidates: {missing}")
    return min(
        grouped,
        key=lambda pair: (
            _mean(grouped[pair], "high_risk_exposures"),
            _mean(grouped[pair], "successful_risk_reposts"),
            _mean(grouped[pair], "community_coverage"),
            _mean(grouped[pair], "benign_exposure_loss"),
            _mean(grouped[pair], "realized_intervention_cost"),
            pair[0],
            pair[1],
        ),
    )


def _condition_accepts_mu(condition: Mapping[str, object], mu: float) -> bool:
    checks = []
    if "mu_min_inclusive" in condition:
        checks.append(mu >= float(condition["mu_min_inclusive"]))
    if "mu_min_exclusive" in condition:
        checks.append(mu > float(condition["mu_min_exclusive"]))
    if "mu_max_inclusive" in condition:
        checks.append(mu <= float(condition["mu_max_inclusive"]))
    if "mu_max_exclusive" in condition:
        checks.append(mu < float(condition["mu_max_exclusive"]))
    return all(checks)


def validate_preregistered_config(config: Mapping[str, Any]) -> dict[str, object]:
    calibration_seeds = [int(seed) for seed in config["calibration_seeds"]]
    evaluation_seeds = [int(seed) for seed in config["evaluation_seeds"]]
    if set(calibration_seeds) & set(evaluation_seeds):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if len(calibration_seeds) < 5 or len(evaluation_seeds) < 10:
        raise ValueError("pilot requires at least 5 calibration and 10 evaluation seeds")
    simulation = config["simulation"]
    if int(simulation["nodes"]) < 60 or int(simulation["timesteps"]) < 8:
        raise ValueError("pilot requires at least 60 nodes and 8 timesteps")
    if tuple(config["baselines"]) != BASELINES:
        raise ValueError("the preregistered five baselines must remain unchanged")
    grid = [tuple(map(float, pair)) for pair in config["omega_grid"]]
    budget = float(config["project_budget"])
    if any(project_l1_cost(*pair) > budget + 1e-12 for pair in grid):
        raise ValueError("an omega candidate exceeds the shared L1 budget")
    baseline_pairs = {
        "no_intervention": (1.0, 1.0),
        "global_throttle": (
            float(config["global_keep_probability"]),
            float(config["global_keep_probability"]),
        ),
        "static_cosref": (
            float(config["static_omega_intra"]),
            float(config["static_omega_inter"]),
        ),
        "dynamic_cosref_initial": (1.0, 1.0),
    }
    over_budget = {
        name: project_l1_cost(*pair)
        for name, pair in baseline_pairs.items()
        if project_l1_cost(*pair) > budget + 1e-12
    }
    if over_budget:
        raise ValueError(f"baseline parameters exceed the shared L1 budget: {over_budget}")

    measurements = []
    seeds = [*calibration_seeds, *evaluation_seeds, int(config["preflight_seed"])]
    for condition in config["conditions"]:
        for seed in seeds:
            communities, edges = generate_directed_sbm(
                nodes=int(simulation["nodes"]),
                p_intra=float(condition["p_intra"]),
                p_inter=float(condition["p_inter"]),
                seed=seed,
            )
            stats = compute_mixing_statistics(edges, communities)
            if stats.mu is None or not _condition_accepts_mu(condition, stats.mu):
                raise ValueError(
                    f"condition {condition['id']} seed {seed} measured mu "
                    f"{stats.mu} outside preregistered interval"
                )
            measurements.append(
                {
                    "condition": str(condition["id"]),
                    "seed": seed,
                    "measured_mu": stats.mu,
                    "phase": (
                        "calibration"
                        if seed in calibration_seeds
                        else "evaluation"
                        if seed in evaluation_seeds
                        else "preflight"
                    ),
                }
            )
    return {
        "valid": True,
        "llm_calls": 0,
        "network_measurements": measurements,
    }


class RobustnessPilot:
    def __init__(
        self,
        config_path: Path,
        output: Path,
        *,
        resume: bool = False,
    ) -> None:
        self.config_path = config_path
        self.output = output
        self.resume = resume
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_sha256 = sha256_file(config_path)
        self.validation = validate_preregistered_config(self.config)

    def _manifest_path(self) -> Path:
        return self.output / "manifest.json"

    def _initialize_output(self, preflight: Mapping[str, object]) -> None:
        if self.output.exists():
            if not self.resume:
                raise FileExistsError(f"refusing to overwrite output: {self.output}")
            manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            if manifest.get("config_sha256") != self.config_sha256:
                raise ValueError("resume refused: preregistered config hash mismatch")
            self._quarantine_pending()
            return
        self.output.mkdir(parents=True)
        (self.output / ".pending").mkdir()
        (self.output / "checkpoints").mkdir()
        (self.output / "quarantine").mkdir()
        shutil.copyfile(self.config_path, self.output / "preregistered_config.json")
        _write_json(
            self._manifest_path(),
            {
                "experiment_id": self.config["experiment_id"],
                "status": "running",
                "config_sha256": self.config_sha256,
                "config_path": str(self.config_path),
                "formal_calibration_runs": (
                    len(self.config["conditions"])
                    * len(self.config["omega_grid"])
                    * len(self.config["calibration_seeds"])
                ),
                "formal_evaluation_runs": (
                    len(self.config["conditions"])
                    * len(self.config["baselines"])
                    * len(self.config["evaluation_seeds"])
                ),
                "preflight": dict(preflight),
                "llm_calls": 0,
            },
        )

    def _quarantine_pending(self) -> None:
        pending_root = self.output / ".pending"
        if not pending_root.exists():
            pending_root.mkdir()
            return
        entries = sorted(pending_root.iterdir())
        if not entries:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = self.output / "quarantine" / stamp
        quarantine.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            shutil.move(str(entry), quarantine / entry.name)

    def _complete_is_valid(self, directory: Path) -> bool:
        complete_path = directory / "complete.json"
        summary_path = directory / "summary.json"
        db_path = directory / "simulation.db"
        if not complete_path.exists() or not summary_path.exists() or not db_path.exists():
            return False
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            complete.get("status") == "completed"
            and complete.get("config_sha256") == self.config_sha256
            and complete.get("summary_sha256") == sha256_file(summary_path)
            and complete.get("database_sha256") == sha256_file(db_path)
        )

    def _quarantine_invalid_final(self, directory: Path) -> None:
        if not directory.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        relative = directory.relative_to(self.output)
        target = self.output / "quarantine" / stamp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), target)

    async def _run_one(
        self,
        *,
        phase: str,
        condition: Mapping[str, Any],
        label: str,
        seed: int,
        baseline: str,
        theory_omega: tuple[float, float] | None,
    ) -> dict[str, object]:
        run_id = f"{phase}--{condition['id']}--{label}--{seed}"
        final = self.output / "raw" / phase / str(condition["id"]) / label / str(seed)
        if self._complete_is_valid(final):
            payload = json.loads((final / "summary.json").read_text(encoding="utf-8"))
            payload["resumed_from_checkpoint"] = True
            return payload
        self._quarantine_invalid_final(final)
        pending = self.output / ".pending" / run_id
        if pending.exists():
            raise RuntimeError(f"pending path unexpectedly exists: {pending}")
        started = time.monotonic()
        result = await run_scenario(
            condition=condition,
            config=self.config,
            baseline=baseline,
            seed=seed,
            output_dir=pending,
            theory_omega=theory_omega,
        )
        duration = time.monotonic() - started
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, final)
        corrected = replace(result, database=str(final / "simulation.db")).as_dict()
        corrected.update(
            {
                "run_id": run_id,
                "phase": phase,
                "status": "completed",
                "duration_seconds": duration,
                "resumed_from_checkpoint": False,
                "parameter_l1_cost_final": project_l1_cost(
                    float(corrected["omega_intra_final"]),
                    float(corrected["omega_inter_final"]),
                ),
                "realized_intervention_cost_per_candidate": (
                    float(corrected["realized_intervention_cost"])
                    / int(corrected["candidate_impressions"])
                    if int(corrected["candidate_impressions"])
                    else 0.0
                ),
            }
        )
        _write_json(final / "summary.json", corrected)
        complete = {
            "status": "completed",
            "run_id": run_id,
            "config_sha256": self.config_sha256,
            "summary_sha256": sha256_file(final / "summary.json"),
            "database_sha256": sha256_file(final / "simulation.db"),
            "duration_seconds": duration,
        }
        _write_json(final / "complete.json", complete)
        return corrected

    def _write_checkpoint(
        self,
        phase: str,
        condition_id: str,
        rows: list[Mapping[str, object]],
        expected: int,
    ) -> None:
        _write_json(
            self.output / "checkpoints" / f"{phase}--{condition_id}.json",
            {
                "phase": phase,
                "condition": condition_id,
                "config_sha256": self.config_sha256,
                "expected_runs": expected,
                "completed_runs": len(rows),
                "complete": len(rows) == expected,
                "run_ids": sorted(str(row["run_id"]) for row in rows),
            },
        )

    def _calibration_analysis(
        self, rows: list[dict[str, object]]
    ) -> tuple[dict[str, object], dict[str, tuple[float, float]]]:
        selection_config = self.config["calibration_selection"]
        resamples = int(selection_config["bootstrap_resamples"])
        confidence = float(self.config["statistics"]["confidence_level"])
        base_seed = int(selection_config["bootstrap_seed"])
        grid = [tuple(map(float, pair)) for pair in self.config["omega_grid"]]
        output: dict[str, object] = {}
        selections: dict[str, tuple[float, float]] = {}
        for condition_index, condition in enumerate(self.config["conditions"]):
            condition_id = str(condition["id"])
            condition_rows = [row for row in rows if row["condition"] == condition_id]
            measured_mu = statistics.fmean(
                float(row["measured_mu"]) for row in condition_rows
            )
            allocation = allocate_cosref_control(
                measured_mu,
                float(self.config["project_budget"]),
                grid,
                tolerance=float(self.config["mu_transition_tolerance"]),
            )
            eligible = {
                (candidate.omega_intra, candidate.omega_inter)
                for candidate in allocation.candidates
                if candidate.direction_consistent
            }
            selected = _select_candidate(condition_rows, eligible)
            selections[condition_id] = selected
            candidate_stats = []
            for pair_index, pair in enumerate(grid):
                pair_rows = [
                    row
                    for row in condition_rows
                    if (
                        float(row["omega_intra_initial"]),
                        float(row["omega_inter_initial"]),
                    )
                    == pair
                ]
                if len(pair_rows) != len(self.config["calibration_seeds"]):
                    raise ValueError(f"candidate {pair} lacks all calibration seeds")
                metrics = {
                    metric: describe_values(
                        [float(row[metric]) for row in pair_rows],
                        resamples=resamples,
                        seed=base_seed + condition_index * 100 + pair_index * 10,
                        confidence=confidence,
                    )
                    for metric in CALIBRATION_METRICS
                }
                candidate_stats.append(
                    {
                        "omega_intra": pair[0],
                        "omega_inter": pair[1],
                        "eligible_for_selection": pair in eligible,
                        "observations": sorted(pair_rows, key=lambda row: int(row["seed"])),
                        "metrics": metrics,
                    }
                )

            calibration_seeds = sorted(map(int, self.config["calibration_seeds"]))
            leave_one_out: Counter[tuple[float, float]] = Counter()
            for omitted in calibration_seeds:
                retained = [row for row in condition_rows if int(row["seed"]) != omitted]
                leave_one_out[_select_candidate(retained, eligible)] += 1

            by_pair_seed = {
                (
                    float(row["omega_intra_initial"]),
                    float(row["omega_inter_initial"]),
                    int(row["seed"]),
                ): row
                for row in condition_rows
            }
            rng = random.Random(base_seed + condition_index)
            bootstrap_choices: Counter[tuple[float, float]] = Counter()
            for _ in range(resamples):
                sampled_seeds = [rng.choice(calibration_seeds) for _ in calibration_seeds]
                sampled_rows = [
                    by_pair_seed[pair[0], pair[1], seed]
                    for pair in grid
                    for seed in sampled_seeds
                ]
                bootstrap_choices[_select_candidate(sampled_rows, eligible)] += 1
            direction_counts: Counter[str] = Counter()
            for pair, count in bootstrap_choices.items():
                direction_counts[_choice_direction(pair)] += count

            eligible_ranked = sorted(
                eligible,
                key=lambda pair: (
                    _mean(
                        [
                            row
                            for row in condition_rows
                            if (
                                float(row["omega_intra_initial"]),
                                float(row["omega_inter_initial"]),
                            )
                            == pair
                        ],
                        "high_risk_exposures",
                    ),
                    pair,
                ),
            )
            best, runner_up = eligible_ranked[:2]

            def candidate_mean(
                pair: tuple[float, float],
                metric: str,
                rows: list[Mapping[str, object]] = condition_rows,
            ) -> float:
                return _mean(
                    [
                        row
                        for row in rows
                        if (
                            float(row["omega_intra_initial"]),
                            float(row["omega_inter_initial"]),
                        )
                        == pair
                    ],
                    metric,
                )

            exact_frequency = bootstrap_choices[selected] / resamples
            selected_direction = _choice_direction(selected)
            direction_frequency = direction_counts[selected_direction] / resamples
            output[condition_id] = {
                "mean_measured_mu": measured_mu,
                "paper_guided_direction": allocation.direction,
                "eligible_candidates": [list(pair) for pair in sorted(eligible)],
                "selected_omega": list(selected),
                "selected_direction": selected_direction,
                "candidate_statistics": candidate_stats,
                "leave_one_out_selection_counts": {
                    f"{pair[0]:.1f},{pair[1]:.1f}": count
                    for pair, count in sorted(leave_one_out.items())
                },
                "bootstrap_selection_frequency": {
                    f"{pair[0]:.1f},{pair[1]:.1f}": count / resamples
                    for pair, count in sorted(bootstrap_choices.items())
                },
                "bootstrap_direction_frequency": {
                    direction: count / resamples
                    for direction, count in sorted(direction_counts.items())
                },
                "selected_exact_frequency": exact_frequency,
                "selected_direction_frequency": direction_frequency,
                "direction_stable": direction_frequency
                >= float(selection_config["direction_stability_threshold"]),
                "exact_omega_stable": exact_frequency
                >= float(selection_config["exact_omega_stability_threshold"]),
                "best_vs_runner_up": {
                    "best": list(best),
                    "runner_up": list(runner_up),
                    "high_risk_exposure_gap_runner_minus_best": candidate_mean(
                        runner_up, "high_risk_exposures"
                    )
                    - candidate_mean(best, "high_risk_exposures"),
                    "risk_repost_gap_runner_minus_best": candidate_mean(
                        runner_up, "successful_risk_reposts"
                    )
                    - candidate_mean(best, "successful_risk_reposts"),
                    "cascade_gap_runner_minus_best": candidate_mean(
                        runner_up, "risk_cascade_size"
                    )
                    - candidate_mean(best, "risk_cascade_size"),
                },
            }
        return output, selections

    def _evaluation_analysis(
        self, rows: list[dict[str, object]]
    ) -> dict[str, object]:
        stats_config = self.config["statistics"]
        resamples = int(stats_config["bootstrap_resamples"])
        confidence = float(stats_config["confidence_level"])
        base_seed = int(stats_config["bootstrap_seed"])
        output: dict[str, object] = {}
        for condition_index, condition in enumerate(self.config["conditions"]):
            condition_id = str(condition["id"])
            condition_rows = [row for row in rows if row["condition"] == condition_id]
            baseline_summary = {}
            for baseline_index, baseline in enumerate(BASELINES):
                baseline_rows = [row for row in condition_rows if row["baseline"] == baseline]
                baseline_summary[baseline] = {
                    metric: describe_values(
                        [float(row[metric]) for row in baseline_rows],
                        resamples=resamples,
                        seed=base_seed
                        + condition_index * 1000
                        + baseline_index * 100
                        + metric_index,
                        confidence=confidence,
                    )
                    for metric_index, metric in enumerate(COMPARISON_METRICS)
                }
            theory_rows = [
                row
                for row in condition_rows
                if row["baseline"] == "theory_informed_cosref"
            ]
            comparisons = {}
            for comparator_index, comparator in enumerate(COMPARATORS):
                comparator_rows = [
                    row for row in condition_rows if row["baseline"] == comparator
                ]
                metric_results = {}
                for metric_index, metric in enumerate(COMPARISON_METRICS):
                    pairs = paired_differences(theory_rows, comparator_rows, metric)
                    values = [float(pair["difference"]) for pair in pairs]
                    description = describe_values(
                        values,
                        resamples=resamples,
                        seed=base_seed
                        + condition_index * 1000
                        + comparator_index * 100
                        + metric_index,
                        confidence=confidence,
                    )
                    low, high = description["bootstrap_ci"]
                    description["interpretation"] = (
                        "stable_decrease"
                        if high < 0
                        else "stable_increase"
                        if low > 0
                        else "uncertain_ci_crosses_zero"
                    )
                    description["paired_differences"] = pairs
                    metric_results[metric] = description
                comparisons[comparator] = metric_results
            output[condition_id] = {
                "actual_mu": describe_values(
                    [
                        float(row["measured_mu"])
                        for row in condition_rows
                        if row["baseline"] == "no_intervention"
                    ],
                    resamples=resamples,
                    seed=base_seed + condition_index,
                    confidence=confidence,
                ),
                "baseline_summary": baseline_summary,
                "paired_comparisons": comparisons,
            }
        return output

    async def preflight(self) -> dict[str, object]:
        condition = next(
            item
            for item in self.config["conditions"]
            if item["id"] == "moderate-mixing"
        )
        with tempfile.TemporaryDirectory(prefix="cosref-preflight-") as directory:
            started = time.monotonic()
            result = await run_scenario(
                condition=condition,
                config=self.config,
                baseline="no_intervention",
                seed=int(self.config["preflight_seed"]),
                output_dir=Path(directory) / "run",
            )
            duration = time.monotonic() - started
        formal_runs = (
            len(self.config["conditions"])
            * len(self.config["omega_grid"])
            * len(self.config["calibration_seeds"])
            + len(self.config["conditions"])
            * len(self.config["baselines"])
            * len(self.config["evaluation_seeds"])
        )
        return {
            "seed": int(self.config["preflight_seed"]),
            "condition": str(condition["id"]),
            "duration_seconds": duration,
            "estimated_formal_runs": formal_runs,
            "estimated_total_seconds": duration * formal_runs,
            "runtime_limit_seconds": int(self.config["maximum_runtime_seconds"]),
            "within_limit": duration * formal_runs
            <= int(self.config["maximum_runtime_seconds"]),
            "measured_mu": result.measured_mu,
            "included_in_formal_results": False,
            "llm_calls": 0,
        }

    async def run(self) -> dict[str, object]:
        wall_started = time.monotonic()
        preflight = await self.preflight()
        if not preflight["within_limit"]:
            return {
                "status": "stopped_runtime_estimate",
                "config_sha256": self.config_sha256,
                "preflight": preflight,
                "llm_calls": 0,
            }
        self._initialize_output(preflight)
        calibration_rows: list[dict[str, object]] = []
        grid = [tuple(map(float, pair)) for pair in self.config["omega_grid"]]
        for condition in self.config["conditions"]:
            condition_rows = []
            for pair in grid:
                label = f"omega-{pair[0]:.1f}-{pair[1]:.1f}"
                for seed in map(int, self.config["calibration_seeds"]):
                    row = await self._run_one(
                        phase="calibration",
                        condition=condition,
                        label=label,
                        seed=seed,
                        baseline="calibration",
                        theory_omega=pair,
                    )
                    calibration_rows.append(row)
                    condition_rows.append(row)
                    self._write_checkpoint(
                        "calibration",
                        str(condition["id"]),
                        condition_rows,
                        len(grid) * len(self.config["calibration_seeds"]),
                    )
        calibration_rows.sort(
            key=lambda row: (
                str(row["condition"]),
                float(row["omega_intra_initial"]),
                float(row["omega_inter_initial"]),
                int(row["seed"]),
            )
        )
        calibration_analysis, selections = self._calibration_analysis(calibration_rows)
        _write_jsonl(self.output / "calibration_observations.jsonl", calibration_rows)
        _write_json(self.output / "calibration_stability.json", calibration_analysis)

        evaluation_rows: list[dict[str, object]] = []
        for condition in self.config["conditions"]:
            condition_id = str(condition["id"])
            condition_rows = []
            for baseline in BASELINES:
                for seed in map(int, self.config["evaluation_seeds"]):
                    row = await self._run_one(
                        phase="evaluation",
                        condition=condition,
                        label=baseline,
                        seed=seed,
                        baseline=baseline,
                        theory_omega=selections[condition_id],
                    )
                    evaluation_rows.append(row)
                    condition_rows.append(row)
                    self._write_checkpoint(
                        "evaluation",
                        condition_id,
                        condition_rows,
                        len(BASELINES) * len(self.config["evaluation_seeds"]),
                    )
        evaluation_rows.sort(
            key=lambda row: (
                str(row["condition"]),
                str(row["baseline"]),
                int(row["seed"]),
            )
        )
        evaluation_analysis = self._evaluation_analysis(evaluation_rows)
        _write_jsonl(self.output / "evaluation_runs.jsonl", evaluation_rows)
        _write_json(self.output / "evaluation_statistics.json", evaluation_analysis)
        total_duration = time.monotonic() - wall_started
        summary = {
            "status": "completed",
            "experiment_id": self.config["experiment_id"],
            "config_sha256": self.config_sha256,
            "preflight": preflight,
            "formal_calibration_runs": len(calibration_rows),
            "formal_evaluation_runs": len(evaluation_rows),
            "total_formal_runs": len(calibration_rows) + len(evaluation_rows),
            "selected_omega": {
                condition: list(pair) for condition, pair in sorted(selections.items())
            },
            "calibration": calibration_analysis,
            "evaluation": evaluation_analysis,
            "network_validation": self.validation,
            "actual_runtime_seconds": total_duration,
            "llm_calls": 0,
            "notes": [
                "No agent behavior or repost rule was changed for this pilot.",
                "Paired differences match ex-ante seeds; downstream events can diverge.",
                "Known synthetic risk labels favor risk-targeted policies on benign loss.",
            ],
        }
        _write_json(self.output / "summary.json", summary)
        manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "completed",
                "actual_runtime_seconds": total_duration,
                "formal_calibration_runs_completed": len(calibration_rows),
                "formal_evaluation_runs_completed": len(evaluation_rows),
                "summary_sha256": sha256_file(self.output / "summary.json"),
                "llm_calls": 0,
            }
        )
        _write_json(self._manifest_path(), manifest)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = RobustnessPilot(args.config, args.output, resume=args.resume)
    result = asyncio.run(pilot.run())
    print(
        json.dumps(
            {
                "status": result["status"],
                "config_sha256": result["config_sha256"],
                "llm_calls": result["llm_calls"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
