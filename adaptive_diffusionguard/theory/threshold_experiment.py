"""Preregistered OASIS threshold-response bridge experiment."""

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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oasis.social_platform.typing import RecsysType

from adaptive_diffusionguard.governance.cosref import (
    NoInterventionPolicy,
    StaticCOSREFPolicy,
)
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

from .allocation import paper_exponential_cost, project_l1_cost
from .experiment import UniformThrottlePolicy, generate_directed_sbm
from .mixing import compute_mixing_statistics
from .robustness import describe_values, paired_differences
from .threshold_response import (
    ThresholdResponseEngine,
    allocate_strict_oasis_keep,
    simulate_paper_native,
    symmetrize_contacts,
)

STRATEGIES = (
    "no_intervention",
    "global_throttle",
    "static_l1",
    "static_cost_matched",
    "theory_informed",
)
COMPARATORS = (
    "no_intervention",
    "global_throttle",
    "static_l1",
    "static_cost_matched",
)
METRICS = (
    "final_risk_adoption_fraction",
    "risk_cascade_size",
    "successful_risk_reposts",
    "high_risk_exposures",
    "intra_risk_adoptions",
    "inter_risk_adoptions",
    "community_coverage",
    "threshold_met_but_exposure_blocked",
    "below_threshold_count",
    "benign_exposure_loss",
    "parameter_l1_cost",
    "realized_intervention_cost",
    "realized_intervention_cost_per_candidate",
    "suppressed_impressions",
    "maximum_risk_post_depth",
    "risk_adoption_generations",
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
    _atomic_write(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _condition_accepts_mu(condition: Mapping[str, object], mu: float) -> bool:
    if "mu_min_inclusive" in condition and mu < float(condition["mu_min_inclusive"]):
        return False
    if "mu_min_exclusive" in condition and mu <= float(condition["mu_min_exclusive"]):
        return False
    if "mu_max_inclusive" in condition and mu > float(condition["mu_max_inclusive"]):
        return False
    return not (
        "mu_max_exclusive" in condition
        and mu >= float(condition["mu_max_exclusive"])
    )


def _contact_edges(
    directed_edges: Iterable[tuple[int, int]], communities: Mapping[int, str]
) -> list[tuple[int, int]]:
    neighbours = symmetrize_contacts(directed_edges, communities)
    return sorted(
        (source, target)
        for source, targets in neighbours.items()
        for target in targets
        if source < target
    )


def _initial_adopters(
    communities: Mapping[int, str], density: float, seed: int, root_index: int
) -> list[int]:
    eligible = sorted(
        user for user, community in communities.items() if community == "community-a"
    )
    count = math.floor(float(density) * len(communities))
    if count <= 0 or count > len(eligible):
        raise ValueError("initial adoption density yields an invalid seeded count")
    return sorted(random.Random(seed + 7000 + root_index).sample(eligible, count))


@dataclass(frozen=True, slots=True)
class ThresholdScenarioResult:
    condition: str
    strategy: str
    seed: int
    measured_mu: float
    allocation_direction: str
    oasis_keep_intra: float
    oasis_keep_inter: float
    paper_omega_intra: float
    paper_omega_inter: float
    threshold: float
    initial_adoption_density: float
    initial_risk_adopters: int
    final_risk_adopters: int
    final_risk_adoption_fraction: float
    successful_risk_reposts: int
    risk_cascade_size: int
    maximum_risk_post_depth: int
    risk_adoption_generations: int
    intra_risk_adoptions: int
    inter_risk_adoptions: int
    community_coverage: float
    threshold_met_but_exposure_blocked: int
    below_threshold_count: int
    candidate_impressions: int
    shown_impressions: int
    suppressed_impressions: int
    high_risk_exposures: int
    benign_exposure_loss: float
    parameter_l1_cost: float
    realized_intervention_cost: float
    realized_intervention_cost_per_candidate: float
    llm_calls: int
    database: str
    audit_path: str
    timestep_path: str
    provenance: str = "project_adaptation_oasis_exposure_gated_threshold"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_threshold_config(config: Mapping[str, Any]) -> dict[str, object]:
    calibration = [int(seed) for seed in config["calibration_seeds"]]
    evaluation = [int(seed) for seed in config["evaluation_seeds"]]
    if set(calibration) & set(evaluation):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if len(calibration) < 5 or len(evaluation) < 10:
        raise ValueError("at least five calibration and ten evaluation seeds required")
    if tuple(config["strategies"]) != STRATEGIES:
        raise ValueError("threshold strategy set changed from preregistration")
    simulation = config["simulation"]
    if int(simulation["nodes"]) < 60 or int(simulation["timesteps"]) < 8:
        raise ValueError("threshold pilot requires at least 60 nodes and 8 timesteps")
    response = config["threshold_response"]
    if (
        float(response["main_paper_omega_intra"]) != 1.0
        or float(response["main_paper_omega_inter"]) != 1.0
    ):
        raise ValueError("main paper omega must remain fixed at (1, 1)")
    if not bool(response["strict_inequality"]) or not bool(response["irreversible"]):
        raise ValueError("paper threshold must remain strict and irreversible")
    grid = [tuple(map(float, pair)) for pair in config["theory_keep_grid"]]
    budget = float(config["project_budget"])
    if any(abs(project_l1_cost(*pair) - budget) > 1e-12 for pair in grid):
        raise ValueError("all theory candidates must use the shared L1 budget")

    seeds = [int(config["preflight_seed"]), *calibration, *evaluation]
    measurements = []
    for condition in config["conditions"]:
        for seed in seeds:
            communities, directed = generate_directed_sbm(
                nodes=int(simulation["nodes"]),
                p_intra=float(condition["p_intra"]),
                p_inter=float(condition["p_inter"]),
                seed=seed,
            )
            contacts = _contact_edges(directed, communities)
            stats = compute_mixing_statistics(
                contacts, communities, directed_input=False
            )
            if stats.mu is None or not _condition_accepts_mu(condition, stats.mu):
                raise ValueError(
                    f"condition {condition['id']} seed {seed} has invalid mu {stats.mu}"
                )
            allocation = allocate_strict_oasis_keep(
                mu=stats.mu,
                project_budget=budget,
                keep_grid=grid,
                tolerance=float(config["direction_tolerance"]),
                minimum_strict_gap=float(config["minimum_strict_gap"]),
            )
            measurements.append(
                {
                    "condition": str(condition["id"]),
                    "seed": seed,
                    "mu": stats.mu,
                    "allocation_direction": allocation.direction,
                    "eligible_oasis_keep_pairs": [
                        list(pair) for pair in allocation.eligible_oasis_keep_pairs
                    ],
                }
            )
    return {"valid": True, "network_measurements": measurements, "llm_calls": 0}


def _policy(
    strategy: str, keep_pair: tuple[float, float], seed: int
) -> StaticCOSREFPolicy:
    if strategy == "no_intervention":
        return NoInterventionPolicy(seed)
    if strategy == "global_throttle":
        return UniformThrottlePolicy(keep_pair[0], seed)
    return StaticCOSREFPolicy(*keep_pair, seed=seed)


def select_cost_matched_static(
    *,
    target_cost: float,
    candidate_mean_costs: Mapping[tuple[float, float], float],
    maximum_relative_error: float,
) -> dict[str, object]:
    """Select the preregistered symmetric keep pair closest in realized cost."""

    if target_cost < 0:
        raise ValueError("target_cost cannot be negative")
    if not candidate_mean_costs:
        raise ValueError("at least one cost-match candidate is required")
    candidates = []
    for pair, raw_cost in candidate_mean_costs.items():
        if not math.isclose(pair[0], pair[1], abs_tol=1e-12):
            raise ValueError("cost-match candidates must be symmetric")
        cost = float(raw_cost)
        relative_error = (
            abs(cost - target_cost) / target_cost
            if target_cost
            else abs(cost - target_cost)
        )
        candidates.append((relative_error, -pair[0], pair, cost))
    relative_error, _, pair, cost = min(candidates)
    return {
        "selected_static_keep": list(pair),
        "target_theory_calibration_mean": target_cost,
        "static_calibration_mean": cost,
        "relative_error": relative_error,
        "within_preregistered_tolerance": relative_error
        <= float(maximum_relative_error),
    }


async def run_threshold_scenario(
    *,
    condition: Mapping[str, Any],
    config: Mapping[str, Any],
    strategy: str,
    seed: int,
    output_dir: Path,
    keep_pair: tuple[float, float],
) -> ThresholdScenarioResult:
    """Run one deterministic-threshold OASIS scenario without any LLM."""

    simulation = config["simulation"]
    response = config["threshold_response"]
    communities, directed = generate_directed_sbm(
        nodes=int(simulation["nodes"]),
        p_intra=float(condition["p_intra"]),
        p_inter=float(condition["p_inter"]),
        seed=seed,
    )
    contacts = _contact_edges(directed, communities)
    mixing = compute_mixing_statistics(contacts, communities, directed_input=False)
    if mixing.mu is None:
        raise ValueError("generated contact graph has no edges")
    allocation = allocate_strict_oasis_keep(
        mu=mixing.mu,
        project_budget=float(config["project_budget"]),
        keep_grid=[tuple(map(float, pair)) for pair in config["theory_keep_grid"]],
        tolerance=float(config["direction_tolerance"]),
        minimum_strict_gap=float(config["minimum_strict_gap"]),
    )
    if strategy == "theory_informed" and keep_pair not in set(
        allocation.eligible_oasis_keep_pairs
    ):
        raise ValueError(
            f"theory pair {keep_pair} is invalid for actual network mu={mixing.mu}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    db_path = output_dir / "simulation.db"
    # OASIS' base recommendation refresh uses the process-global random module.
    # Reset it for every strategy/seed cell so paired runs start identically.
    random.seed(seed)
    platform = AdaptiveDiffusionPlatform(
        str(db_path),
        user_communities=communities,
        post_risk_scores={},
        policy=_policy(strategy, keep_pair, seed),
        random_seed=seed,
        run_id=f"threshold-{condition['id']}-{strategy}-{seed}",
        intervention_budget=(
            None if strategy == "global_throttle" else max(2.0, float(config["project_budget"]))
        ),
        recsys_type=RecsysType.RANDOM,
        refresh_rec_post_count=int(simulation["recommendation_count"]),
        following_post_count=int(simulation["following_count"]),
        max_rec_post_len=int(simulation["recommendation_buffer"]),
    )
    for user_id, community in sorted(communities.items()):
        platform.db.execute(
            """
            INSERT INTO user
            (user_id, agent_id, user_name, name, bio, created_at,
             num_followings, num_followers)
            VALUES (?, ?, ?, ?, ?, 0, 0, 0)
            """,
            (
                user_id,
                user_id,
                f"threshold-user-{user_id}",
                f"Synthetic Threshold User {user_id}",
                f"synthetic profile in {community}",
            ),
        )
    bidirectional = [arc for edge in contacts for arc in (edge, (edge[1], edge[0]))]
    platform.db.executemany(
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, 0)",
        bidirectional,
    )
    platform.db.commit()

    density = float(response["initial_adoption_density"])
    root_specs = (
        ("risk", "Synthetic high-risk threshold claim", float(simulation["risk_score"])),
        ("benign", "Synthetic verified benign threshold update", 0.0),
    )
    roots: dict[str, int] = {}
    initial_by_root: dict[int, list[int]] = {}
    for root_index, (kind, content, risk) in enumerate(root_specs):
        initial = _initial_adopters(communities, density, seed, root_index)
        created = await platform.create_post(initial[0], content)
        root_id = int(created["post_id"])
        roots[kind] = root_id
        platform.post_risk_scores[root_id] = risk
        for user_id in initial[1:]:
            repost = await platform.repost(user_id, root_id)
            if not repost.get("success"):
                raise RuntimeError("failed to materialize an initial adopter")
        initial_by_root[root_id] = initial
    _write_json(
        output_dir / "initialization.json",
        {
            "seed": seed,
            "initial_adoption_density": density,
            "initialization_method": response["initialization"],
            "initial_adopters_by_root": {
                str(root): users for root, users in sorted(initial_by_root.items())
            },
            "root_kinds": {str(root): kind for kind, root in roots.items()},
            "paper_omega_intra": float(response["main_paper_omega_intra"]),
            "paper_omega_inter": float(response["main_paper_omega_inter"]),
            "oasis_keep_intra": keep_pair[0],
            "oasis_keep_inter": keep_pair[1],
            "control_path": "oasis_exposure_then_fixed_paper_threshold",
        },
    )
    engine = ThresholdResponseEngine(
        communities=communities,
        contact_edges=contacts,
        initial_adopters=initial_by_root,
        threshold=float(response["threshold"]),
        paper_omega_intra=float(response["main_paper_omega_intra"]),
        paper_omega_inter=float(response["main_paper_omega_inter"]),
        exposure_gate_enabled=True,
    )
    root_author_community = {
        root: communities[users[0]] for root, users in initial_by_root.items()
    }
    audits: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    risk_root = roots["risk"]
    initial_risk_count = len(initial_by_root[risk_root])
    successful_risk_reposts = 0

    for timestep in range(1, int(simulation["timesteps"]) + 1):
        platform.sandbox_clock.time_step = timestep
        await platform.update_rec_table()
        snapshot = engine.begin_timestep(timestep)
        pending: list[tuple[int, int, int, int]] = []
        timestep_audits: list[dict[str, object]] = []
        for user_id in sorted(communities):
            refresh = await platform.refresh(user_id)
            visible_targets: dict[int, list[int]] = defaultdict(list)
            observable_authors: dict[int, set[int]] = defaultdict(set)
            for post in refresh.get("posts", []):
                post_id = int(post["post_id"])
                root_id, _ = platform._root_post(post_id)
                direct_author = int(post["user_id"])
                visible_targets[root_id].append(post_id)
                if direct_author in snapshot.adopted_by_root.get(root_id, frozenset()):
                    observable_authors[root_id].add(direct_author)
            for root_id in sorted(initial_by_root):
                evaluation = engine.evaluate(
                    user_id=user_id,
                    root_post_id=root_id,
                    root_author_community=root_author_community[root_id],
                    observable_adopter_ids=observable_authors[root_id],
                )
                audit = evaluation.as_dict()
                audit.update(
                    {
                        "condition": str(condition["id"]),
                        "strategy": strategy,
                        "seed": seed,
                        "risk_score": platform.post_risk_scores[root_id],
                        "oasis_keep_intra": keep_pair[0],
                        "oasis_keep_inter": keep_pair[1],
                        "target_post_id": None,
                        "failure_category": None,
                    }
                )
                audit_index = len(timestep_audits)
                timestep_audits.append(audit)
                if evaluation.should_attempt_adoption:
                    targets = visible_targets[root_id]
                    if not targets:
                        audit["failure_category"] = "threshold_met_without_visible_target"
                    else:
                        target = max(targets)
                        audit["target_post_id"] = target
                        pending.append((user_id, root_id, target, audit_index))

        successful: list[tuple[int, int]] = []
        for user_id, root_id, target, audit_index in sorted(pending):
            result = await platform.repost(user_id, target)
            success = bool(result.get("success"))
            timestep_audits[audit_index]["dispatcher_success"] = success
            timestep_audits[audit_index]["final_adopted"] = success
            if success:
                successful.append((user_id, root_id))
                if root_id == risk_root:
                    successful_risk_reposts += 1
            else:
                timestep_audits[audit_index]["failure_category"] = "oasis_repost_failed"
        committed = engine.commit(timestep, successful)
        if set(committed) != set(successful):
            raise RuntimeError("threshold state and OASIS dispatcher diverged")
        for audit in timestep_audits:
            if audit["already_adopted"]:
                audit["final_adopted"] = True
        audits.extend(timestep_audits)
        new_risk = sum(root == risk_root for _, root in committed)
        timestep_rows.append(
            {
                "timestep": timestep,
                "frozen_risk_adopters": len(snapshot.adopted_by_root[risk_root]),
                "new_risk_adopters": new_risk,
                "total_risk_adopters": len(engine.adopters(risk_root)),
                "new_total_adopters": len(committed),
                "threshold_met_but_exposure_blocked": sum(
                    bool(row["threshold_met_but_exposure_blocked"])
                    and not bool(row["already_adopted"])
                    for row in timestep_audits
                ),
            }
        )

    audit_path = output_dir / "threshold_decisions.jsonl"
    timestep_path = output_dir / "timestep_adoption.jsonl"
    _write_jsonl(audit_path, audits)
    _write_jsonl(timestep_path, timestep_rows)
    impressions = platform.db.execute(
        """
        SELECT user_community, author_community, risk_score,
               keep_probability, shown
        FROM diffusionguard_impression WHERE run_id = ?
        """,
        (platform.run_id,),
    ).fetchall()
    risky = [row for row in impressions if float(row[2]) > 0]
    benign = [row for row in impressions if float(row[2]) == 0]
    risk_posts = []
    maximum_depth = 0
    for post_id, parent_id in platform.db.execute(
        "SELECT post_id, original_post_id FROM post"
    ).fetchall():
        root_id, _ = platform._root_post(int(post_id))
        if root_id != risk_root:
            continue
        risk_posts.append(int(post_id))
        depth = 0
        current = parent_id
        while current is not None:
            depth += 1
            current_row = platform.db.execute(
                "SELECT original_post_id FROM post WHERE post_id = ?", (current,)
            ).fetchone()
            current = None if current_row is None else current_row[0]
        maximum_depth = max(maximum_depth, depth)
    risk_adopters = engine.adopters(risk_root)
    risk_author_community = root_author_community[risk_root]
    candidate_count = len(impressions)
    realized_cost = sum(1.0 - float(row[3]) for row in impressions)
    result = ThresholdScenarioResult(
        condition=str(condition["id"]),
        strategy=strategy,
        seed=seed,
        measured_mu=float(mixing.mu),
        allocation_direction=allocation.direction,
        oasis_keep_intra=keep_pair[0],
        oasis_keep_inter=keep_pair[1],
        paper_omega_intra=float(response["main_paper_omega_intra"]),
        paper_omega_inter=float(response["main_paper_omega_inter"]),
        threshold=float(response["threshold"]),
        initial_adoption_density=density,
        initial_risk_adopters=initial_risk_count,
        final_risk_adopters=len(risk_adopters),
        final_risk_adoption_fraction=len(risk_adopters) / len(communities),
        successful_risk_reposts=successful_risk_reposts,
        risk_cascade_size=len(risk_posts),
        maximum_risk_post_depth=maximum_depth,
        risk_adoption_generations=sum(
            int(row["new_risk_adopters"]) > 0 for row in timestep_rows
        ),
        intra_risk_adoptions=sum(
            communities[user] == risk_author_community for user in risk_adopters
        ),
        inter_risk_adoptions=sum(
            communities[user] != risk_author_community for user in risk_adopters
        ),
        community_coverage=(
            len({communities[user] for user in risk_adopters})
            / len(set(communities.values()))
        ),
        threshold_met_but_exposure_blocked=sum(
            bool(row["threshold_met_but_exposure_blocked"])
            and not bool(row["already_adopted"])
            for row in audits
        ),
        below_threshold_count=sum(
            not bool(row["threshold_satisfied"]) and not bool(row["already_adopted"])
            for row in audits
        ),
        candidate_impressions=candidate_count,
        shown_impressions=sum(bool(row[4]) for row in impressions),
        suppressed_impressions=sum(not bool(row[4]) for row in impressions),
        high_risk_exposures=sum(bool(row[4]) for row in risky),
        benign_exposure_loss=(
            sum(not bool(row[4]) for row in benign) / len(benign) if benign else 0.0
        ),
        parameter_l1_cost=project_l1_cost(*keep_pair),
        realized_intervention_cost=realized_cost,
        realized_intervention_cost_per_candidate=(
            realized_cost / candidate_count if candidate_count else 0.0
        ),
        llm_calls=0,
        database=str(db_path),
        audit_path=str(audit_path),
        timestep_path=str(timestep_path),
    )
    _write_json(output_dir / "summary.json", result.as_dict())
    platform.db_cursor.close()
    platform.db.close()
    return result


class ThresholdPilot:
    def __init__(self, config_path: Path, output: Path, *, resume: bool = False) -> None:
        self.config_path = config_path
        self.output = output
        self.resume = resume
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_sha256 = sha256_file(config_path)
        implementation_digest = hashlib.sha256()
        for path in sorted(
            (Path(__file__), Path(__file__).with_name("threshold_response.py"))
        ):
            implementation_digest.update(path.name.encode("utf-8"))
            implementation_digest.update(path.read_bytes())
        self.implementation_sha256 = implementation_digest.hexdigest()
        self.validation = validate_threshold_config(self.config)

    def _manifest_path(self) -> Path:
        return self.output / "manifest.json"

    def _initialize_output(self, preflight: Mapping[str, object]) -> None:
        if self.output.exists():
            if not self.resume:
                raise FileExistsError(f"refusing to overwrite output: {self.output}")
            manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            if manifest.get("config_sha256") != self.config_sha256:
                raise ValueError("resume refused: preregistered config hash mismatch")
            stored_implementation = manifest.get("implementation_sha256")
            if (
                stored_implementation is not None
                and stored_implementation != self.implementation_sha256
            ):
                raise ValueError("resume refused: implementation hash mismatch")
            if stored_implementation is None:
                manifest["implementation_sha256"] = self.implementation_sha256
                _write_json(self._manifest_path(), manifest)
            self._quarantine_pending()
            return
        self.output.mkdir(parents=True)
        for name in (".pending", "checkpoints", "quarantine"):
            (self.output / name).mkdir()
        shutil.copyfile(self.config_path, self.output / "preregistered_config.json")
        _write_json(
            self._manifest_path(),
            {
                "experiment_id": self.config["experiment_id"],
                "status": "running",
                "config_sha256": self.config_sha256,
                "implementation_sha256": self.implementation_sha256,
                "preflight": dict(preflight),
                "llm_calls": 0,
            },
        )

    def _quarantine_pending(self) -> None:
        pending = self.output / ".pending"
        pending.mkdir(exist_ok=True)
        entries = sorted(pending.iterdir())
        if not entries:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.output / "quarantine" / stamp
        target.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            shutil.move(str(entry), target / entry.name)

    def _complete_valid(self, directory: Path) -> bool:
        complete_path = directory / "complete.json"
        summary_path = directory / "summary.json"
        db_path = directory / "simulation.db"
        if not all(path.exists() for path in (complete_path, summary_path, db_path)):
            return False
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            complete.get("status") == "completed"
            and complete.get("config_sha256") == self.config_sha256
            and complete.get("implementation_sha256") == self.implementation_sha256
            and complete.get("summary_sha256") == sha256_file(summary_path)
            and complete.get("database_sha256") == sha256_file(db_path)
        )

    def _quarantine_invalid(self, directory: Path) -> None:
        if not directory.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.output / "quarantine" / stamp / directory.relative_to(self.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), target)

    async def _run_one(
        self,
        *,
        phase: str,
        condition: Mapping[str, Any],
        label: str,
        strategy: str,
        seed: int,
        keep_pair: tuple[float, float],
    ) -> dict[str, object]:
        run_id = f"{phase}--{condition['id']}--{label}--{seed}"
        final = self.output / "raw" / phase / str(condition["id"]) / label / str(seed)
        if self._complete_valid(final):
            payload = json.loads((final / "summary.json").read_text(encoding="utf-8"))
            payload["resumed_from_checkpoint"] = True
            return payload
        self._quarantine_invalid(final)
        pending = self.output / ".pending" / run_id
        started = time.monotonic()
        result = await run_threshold_scenario(
            condition=condition,
            config=self.config,
            strategy=strategy,
            seed=seed,
            output_dir=pending,
            keep_pair=keep_pair,
        )
        duration = time.monotonic() - started
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, final)
        payload = replace(
            result,
            database=str(final / "simulation.db"),
            audit_path=str(final / "threshold_decisions.jsonl"),
            timestep_path=str(final / "timestep_adoption.jsonl"),
        ).as_dict()
        payload.update(
            {
                "run_id": run_id,
                "phase": phase,
                "status": "completed",
                "duration_seconds": duration,
                "resumed_from_checkpoint": False,
            }
        )
        _write_json(final / "summary.json", payload)
        _write_json(
            final / "complete.json",
            {
                "status": "completed",
                "run_id": run_id,
                "config_sha256": self.config_sha256,
                "implementation_sha256": self.implementation_sha256,
                "summary_sha256": sha256_file(final / "summary.json"),
                "database_sha256": sha256_file(final / "simulation.db"),
                "duration_seconds": duration,
            },
        )
        return payload

    def _checkpoint(
        self, phase: str, condition: str, rows: list[Mapping[str, object]], expected: int
    ) -> None:
        _write_json(
            self.output / "checkpoints" / f"{phase}--{condition}.json",
            {
                "phase": phase,
                "condition": condition,
                "config_sha256": self.config_sha256,
                "expected_runs": expected,
                "completed_runs": len(rows),
                "complete": len(rows) == expected,
                "run_ids": sorted(str(row["run_id"]) for row in rows),
            },
        )

    async def preflight(self) -> dict[str, object]:
        condition = next(
            row for row in self.config["conditions"] if row["id"] == "moderate-mixing"
        )
        with tempfile.TemporaryDirectory(prefix="threshold-preflight-") as directory:
            started = time.monotonic()
            result = await run_threshold_scenario(
                condition=condition,
                config=self.config,
                strategy="no_intervention",
                seed=int(self.config["preflight_seed"]),
                output_dir=Path(directory) / "run",
                keep_pair=(1.0, 1.0),
            )
            duration = time.monotonic() - started
        theory_grid = {tuple(map(float, pair)) for pair in self.config["theory_keep_grid"]}
        cost_grid = {
            tuple(map(float, pair)) for pair in self.config["static_cost_match_grid"]
        }
        calibration_runs = (
            len(self.config["conditions"])
            * len(theory_grid | cost_grid)
            * len(self.config["calibration_seeds"])
        )
        evaluation_runs = (
            len(self.config["conditions"])
            * len(STRATEGIES)
            * len(self.config["evaluation_seeds"])
        )
        estimate = duration * (calibration_runs + evaluation_runs)
        return {
            "seed": int(self.config["preflight_seed"]),
            "condition": str(condition["id"]),
            "duration_seconds": duration,
            "measured_mu": result.measured_mu,
            "estimated_calibration_runs": calibration_runs,
            "estimated_evaluation_runs": evaluation_runs,
            "estimated_total_seconds": estimate,
            "runtime_limit_seconds": int(self.config["maximum_runtime_seconds"]),
            "within_limit": estimate <= int(self.config["maximum_runtime_seconds"]),
            "included_in_formal_results": False,
            "llm_calls": 0,
        }

    def _paper_native_check(self) -> dict[str, object]:
        spec = self.config["paper_native_check"]
        step = float(spec["omega_step"])
        grid = [index * step for index in range(round(1.0 / step) + 1)]
        raw = []
        optima = []
        for condition in self.config["conditions"]:
            if condition["id"] not in spec["conditions"]:
                continue
            for omega_intra in grid:
                for omega_inter in grid:
                    outcomes = []
                    for seed in map(int, self.config["paper_native_seeds"]):
                        communities, directed = generate_directed_sbm(
                            nodes=int(spec["nodes"]),
                            p_intra=float(condition["p_intra"]),
                            p_inter=float(condition["p_inter"]),
                            seed=seed,
                        )
                        contacts = _contact_edges(directed, communities)
                        initial = _initial_adopters(
                            communities,
                            float(spec["initial_adoption_density"]),
                            seed,
                            0,
                        )
                        result = simulate_paper_native(
                            communities=communities,
                            contact_edges=contacts,
                            initial_adopters=initial,
                            threshold=float(spec["threshold"]),
                            paper_omega_intra=omega_intra,
                            paper_omega_inter=omega_inter,
                            timesteps=int(spec["timesteps"]),
                        )
                        outcomes.append(float(result["final_adoption_fraction"]))
                    raw.append(
                        {
                            "condition": str(condition["id"]),
                            "paper_omega_intra": omega_intra,
                            "paper_omega_inter": omega_inter,
                            "mean_final_adoption_fraction": statistics.fmean(outcomes),
                            "observations": outcomes,
                            "paper_cost": paper_exponential_cost(
                                omega_intra, omega_inter
                            ),
                        }
                    )
            eligible = [
                row
                for row in raw
                if row["condition"] == condition["id"]
                and float(row["mean_final_adoption_fraction"])
                <= float(spec["non_diffusion_cutoff"])
            ]
            if not eligible:
                optima.append({"condition": condition["id"], "available": False})
                continue
            optimum = min(
                eligible,
                key=lambda row: (
                    float(row["paper_cost"]),
                    float(row["paper_omega_intra"]),
                    float(row["paper_omega_inter"]),
                ),
            )
            expected = "intra" if condition["id"] == "strong-community" else "inter"
            observed = (
                "intra"
                if float(optimum["paper_omega_intra"])
                < float(optimum["paper_omega_inter"])
                else "inter"
                if float(optimum["paper_omega_inter"])
                < float(optimum["paper_omega_intra"])
                else "balanced"
            )
            optima.append(
                {
                    **optimum,
                    "available": True,
                    "expected_direction": expected,
                    "observed_direction": observed,
                    "direction_passed": observed == expected,
                }
            )
        return {
            "status": "passed"
            if optima and all(row.get("direction_passed") for row in optima)
            else "failed",
            "exposure_gate_enabled": False,
            "update_mode": "synchronous",
            "strict_inequality": True,
            "raw_grid": raw,
            "optima": optima,
            "official_reference": {
                "commit": "9629dd7a7adecaadfffd53f2ae0f3a28a75a54eb",
                "strong_grid_optimum": [0.0, 0.8],
                "weak_grid_optimum": [0.8, 0.0],
                "source": "runs/cosref-reference-validation/reference-oracle/summary.json",
            },
            "llm_calls": 0,
        }

    @staticmethod
    def _select_theory(
        rows: list[Mapping[str, object]], eligible: set[tuple[float, float]]
    ) -> tuple[float, float]:
        grouped: dict[tuple[float, float], list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            pair = (float(row["oasis_keep_intra"]), float(row["oasis_keep_inter"]))
            if pair in eligible:
                grouped[pair].append(row)
        if set(grouped) != eligible:
            raise ValueError("eligible theory candidates lack calibration observations")
        return min(
            grouped,
            key=lambda pair: (
                statistics.fmean(
                    float(row["final_risk_adoption_fraction"])
                    for row in grouped[pair]
                ),
                statistics.fmean(float(row["risk_cascade_size"]) for row in grouped[pair]),
                statistics.fmean(
                    float(row["successful_risk_reposts"]) for row in grouped[pair]
                ),
                statistics.fmean(
                    float(row["realized_intervention_cost"]) for row in grouped[pair]
                ),
                pair,
            ),
        )

    def _calibration_analysis(
        self, rows: list[dict[str, object]]
    ) -> tuple[
        dict[str, object],
        dict[str, dict[str, tuple[float, float]]],
        dict[str, dict[str, tuple[float, float]]],
    ]:
        config = self.config["calibration_selection"]
        resamples = int(config["bootstrap_resamples"])
        bootstrap_seed = int(config["bootstrap_seed"])
        confidence = float(self.config["statistics"]["confidence_level"])
        theory_grid = {tuple(map(float, pair)) for pair in self.config["theory_keep_grid"]}
        cost_grid = {tuple(map(float, pair)) for pair in self.config["static_cost_match_grid"]}
        output = {}
        selected_theory = {}
        selected_cost = {}
        for condition_index, condition in enumerate(self.config["conditions"]):
            condition_id = str(condition["id"])
            condition_rows = [row for row in rows if row["condition"] == condition_id]
            seed_rows = {
                int(row["seed"]): row
                for row in condition_rows
                if (float(row["oasis_keep_intra"]), float(row["oasis_keep_inter"]))
                == min(theory_grid)
            }
            allocations = {
                seed: allocate_strict_oasis_keep(
                    mu=float(row["measured_mu"]),
                    project_budget=float(self.config["project_budget"]),
                    keep_grid=theory_grid,
                    tolerance=float(self.config["direction_tolerance"]),
                    minimum_strict_gap=float(self.config["minimum_strict_gap"]),
                )
                for seed, row in seed_rows.items()
            }
            seeds_by_direction: dict[str, list[int]] = defaultdict(list)
            for seed, allocation in allocations.items():
                seeds_by_direction[allocation.direction].append(seed)
            by_pair_seed = {
                (
                    float(row["oasis_keep_intra"]),
                    float(row["oasis_keep_inter"]),
                    int(row["seed"]),
                ): row
                for row in condition_rows
            }
            direction_results = {}
            selected_theory[condition_id] = {}
            selected_cost[condition_id] = {}
            for direction_index, (direction, direction_seeds) in enumerate(
                sorted(seeds_by_direction.items())
            ):
                direction_seeds = sorted(direction_seeds)
                eligible = set(
                    allocations[direction_seeds[0]].eligible_oasis_keep_pairs
                )
                direction_rows = [
                    row for row in condition_rows if int(row["seed"]) in direction_seeds
                ]
                selected = self._select_theory(direction_rows, eligible)
                selected_theory[condition_id][direction] = selected
                theory_rows = [
                    row
                    for row in direction_rows
                    if (
                        float(row["oasis_keep_intra"]),
                        float(row["oasis_keep_inter"]),
                    )
                    == selected
                ]
                target_cost = statistics.fmean(
                    float(row["realized_intervention_cost"]) for row in theory_rows
                )
                candidate_mean_costs = {}
                for pair in sorted(cost_grid):
                    pair_rows = [
                        row
                        for row in direction_rows
                        if (
                            float(row["oasis_keep_intra"]),
                            float(row["oasis_keep_inter"]),
                        )
                        == pair
                    ]
                    candidate_mean_costs[pair] = statistics.fmean(
                        float(row["realized_intervention_cost"]) for row in pair_rows
                    )
                cost_match = select_cost_matched_static(
                    target_cost=target_cost,
                    candidate_mean_costs=candidate_mean_costs,
                    maximum_relative_error=float(
                        self.config["cost_matching"]["maximum_relative_error"]
                    ),
                )
                selected_cost[condition_id][direction] = tuple(
                    map(float, cost_match["selected_static_keep"])
                )
                loo: Counter[tuple[float, float]] = Counter()
                if len(direction_seeds) > 1:
                    for omitted in direction_seeds:
                        retained = [seed for seed in direction_seeds if seed != omitted]
                        loo[self._select_theory(
                            [
                                row
                                for row in direction_rows
                                if int(row["seed"]) in retained
                            ],
                            eligible,
                        )] += 1
                rng = random.Random(
                    bootstrap_seed + condition_index * 10 + direction_index
                )
                bootstrap: Counter[tuple[float, float]] = Counter()
                for _ in range(resamples):
                    sampled = [rng.choice(direction_seeds) for _ in direction_seeds]
                    sample_rows = [
                        by_pair_seed[pair[0], pair[1], seed]
                        for pair in theory_grid | cost_grid
                        for seed in sampled
                    ]
                    bootstrap[self._select_theory(sample_rows, eligible)] += 1
                direction_results[direction] = {
                    "calibration_seeds": direction_seeds,
                    "strict_eligible_candidates": [
                        list(pair) for pair in sorted(eligible)
                    ],
                    "selected_theory_keep": list(selected),
                    "leave_one_out_selection_counts": {
                        f"{pair[0]:.1f},{pair[1]:.1f}": count
                        for pair, count in sorted(loo.items())
                    },
                    "bootstrap_selection_frequency": {
                        f"{pair[0]:.1f},{pair[1]:.1f}": count / resamples
                        for pair, count in sorted(bootstrap.items())
                    },
                    "direction_selection_frequency": 1.0,
                    "exact_selection_frequency": bootstrap[selected] / resamples,
                    "direction_stable": True,
                    "exact_parameter_stable": bootstrap[selected] / resamples
                    >= float(config["exact_parameter_stability_threshold"]),
                    "cost_matching": cost_match,
                    "limited_calibration_seeds": len(direction_seeds) < 5,
                }
            candidate_statistics = []
            for pair_index, pair in enumerate(sorted(theory_grid | cost_grid)):
                pair_rows = [
                    row
                    for row in condition_rows
                    if (float(row["oasis_keep_intra"]), float(row["oasis_keep_inter"]))
                    == pair
                ]
                candidate_statistics.append(
                    {
                        "oasis_keep_intra": pair[0],
                        "oasis_keep_inter": pair[1],
                        "strict_theory_eligible_by_seed": {
                            str(seed): pair
                            in set(allocation.eligible_oasis_keep_pairs)
                            for seed, allocation in sorted(allocations.items())
                        },
                        "static_cost_match_candidate": pair in cost_grid,
                        "observations": sorted(pair_rows, key=lambda row: int(row["seed"])),
                        "metrics": {
                            metric: describe_values(
                                [float(row[metric]) for row in pair_rows],
                                resamples=resamples,
                                seed=bootstrap_seed
                                + condition_index * 100
                                + pair_index * 10
                                + metric_index,
                                confidence=confidence,
                            )
                            for metric_index, metric in enumerate(METRICS)
                        },
                    }
                )
            output[condition_id] = {
                "actual_mu_by_seed": {
                    str(seed): row["measured_mu"] for seed, row in sorted(seed_rows.items())
                },
                "allocation_direction_by_seed": {
                    str(seed): allocation.direction
                    for seed, allocation in sorted(allocations.items())
                },
                "selection_by_actual_mu_direction": direction_results,
                "candidate_statistics": candidate_statistics,
            }
        return output, selected_theory, selected_cost

    def _evaluation_analysis(self, rows: list[dict[str, object]]) -> dict[str, object]:
        statistics_config = self.config["statistics"]
        resamples = int(statistics_config["bootstrap_resamples"])
        base_seed = int(statistics_config["bootstrap_seed"])
        confidence = float(statistics_config["confidence_level"])
        output = {}
        for condition_index, condition in enumerate(self.config["conditions"]):
            condition_id = str(condition["id"])
            condition_rows = [row for row in rows if row["condition"] == condition_id]
            absolute = {}
            for strategy_index, strategy in enumerate(STRATEGIES):
                strategy_rows = [
                    row for row in condition_rows if row["strategy"] == strategy
                ]
                absolute[strategy] = {
                    metric: describe_values(
                        [float(row[metric]) for row in strategy_rows],
                        resamples=resamples,
                        seed=base_seed
                        + condition_index * 1000
                        + strategy_index * 100
                        + metric_index,
                        confidence=confidence,
                    )
                    for metric_index, metric in enumerate(METRICS)
                }
            theory_rows = [
                row for row in condition_rows if row["strategy"] == "theory_informed"
            ]
            comparisons = {}
            for comparator_index, comparator in enumerate(COMPARATORS):
                comparator_rows = [
                    row for row in condition_rows if row["strategy"] == comparator
                ]
                comparisons[comparator] = {}
                for metric_index, metric in enumerate(METRICS):
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
                    comparisons[comparator][metric] = description
            output[condition_id] = {
                "actual_mu": describe_values(
                    [
                        float(row["measured_mu"])
                        for row in condition_rows
                        if row["strategy"] == "no_intervention"
                    ],
                    resamples=resamples,
                    seed=base_seed + condition_index,
                    confidence=confidence,
                ),
                "absolute": absolute,
                "paired_comparisons": comparisons,
            }
        return output

    def _write_report(self, summary: Mapping[str, Any]) -> None:
        lines = [
            "# COSREF threshold-response bridge v1 实验报告",
            "",
            "本报告来自无 LLM 的合成网络机制实验，不是人类行为模型或因果治理结论。",
            "paper omega 与 OASIS keep probability 在代码、配置和结果中保持分离。",
            "",
            "## 完整性",
            "",
            f"- 配置 SHA-256：`{self.config_sha256}`",
            f"- paper-native 对照：`{summary['paper_native']['status']}`",
            f"- 校准运行：{summary['formal_calibration_runs']}",
            f"- 评估运行：{summary['formal_evaluation_runs']}",
            f"- 实际运行时间：{summary['actual_runtime_seconds']:.3f} 秒",
            "- 远程 LLM 调用：0",
            "",
            "## 校准与成本匹配",
            "",
        ]
        for condition, result in sorted(summary["calibration"].items()):
            for direction, selection in sorted(
                result["selection_by_actual_mu_direction"].items()
            ):
                match = selection["cost_matching"]
                lines.append(
                    f"- {condition}/{direction}: "
                    f"theory keep={selection['selected_theory_keep']}, "
                    f"exact frequency={selection['exact_selection_frequency']:.3f}, "
                    f"cost-matched static={match['selected_static_keep']}, "
                    f"calibration relative error={match['relative_error']:.3f}."
                )
        lines.extend(
            [
                "",
                "## 限制",
                "",
                "- 60 节点、两个合成根帖子、有限种子，不代表真实人群。",
                "- 风险标签为实验真值，正常内容损失比较偏向风险定向策略。",
                "- OASIS feed 使邻居采纳变为可观察信号是项目适配，不是论文公式。",
                "- 固定种子 bootstrap 描述本实验种子变异，不提供外部有效性保证。",
                "- 区间跨越零的传播差异必须解释为不确定。",
                "",
                "完整逐种子差值和区间见 `evaluation_statistics.json`。",
            ]
        )
        _atomic_write(self.output / "实验报告.md", "\n".join(lines) + "\n")

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
        paper_native = self._paper_native_check()
        _write_json(self.output / "paper_native_check.json", paper_native)
        if paper_native["status"] != "passed":
            summary = {
                "status": "failed_paper_native_check",
                "config_sha256": self.config_sha256,
                "preflight": preflight,
                "paper_native": paper_native,
                "llm_calls": 0,
            }
            _write_json(self.output / "summary.json", summary)
            return summary

        theory_grid = {tuple(map(float, pair)) for pair in self.config["theory_keep_grid"]}
        cost_grid = {
            tuple(map(float, pair)) for pair in self.config["static_cost_match_grid"]
        }
        calibration_pairs = sorted(theory_grid | cost_grid)
        calibration_rows: list[dict[str, object]] = []
        for condition in self.config["conditions"]:
            condition_rows = []
            for pair in calibration_pairs:
                label = f"keep-{pair[0]:.1f}-{pair[1]:.1f}"
                for seed in map(int, self.config["calibration_seeds"]):
                    row = await self._run_one(
                        phase="calibration",
                        condition=condition,
                        label=label,
                        strategy="calibration",
                        seed=seed,
                        keep_pair=pair,
                    )
                    calibration_rows.append(row)
                    condition_rows.append(row)
                    self._checkpoint(
                        "calibration",
                        str(condition["id"]),
                        condition_rows,
                        len(calibration_pairs) * len(self.config["calibration_seeds"]),
                    )
        calibration_rows.sort(
            key=lambda row: (
                str(row["condition"]),
                float(row["oasis_keep_intra"]),
                float(row["oasis_keep_inter"]),
                int(row["seed"]),
            )
        )
        calibration, selected_theory, selected_cost = self._calibration_analysis(
            calibration_rows
        )
        _write_jsonl(self.output / "calibration_runs.jsonl", calibration_rows)
        _write_json(self.output / "calibration_stability.json", calibration)
        _write_json(
            self.output / "cost_matching.json",
            {
                condition: {
                    direction: selection["cost_matching"]
                    for direction, selection in sorted(
                        result["selection_by_actual_mu_direction"].items()
                    )
                }
                for condition, result in sorted(calibration.items())
            },
        )

        evaluation_rows: list[dict[str, object]] = []
        for condition in self.config["conditions"]:
            condition_id = str(condition["id"])
            condition_rows = []
            for strategy in STRATEGIES:
                for seed in map(int, self.config["evaluation_seeds"]):
                    measurement = next(
                        item
                        for item in self.validation["network_measurements"]
                        if item["condition"] == condition_id
                        and int(item["seed"]) == seed
                    )
                    direction = str(measurement["allocation_direction"])
                    if strategy == "no_intervention":
                        pair = (1.0, 1.0)
                    elif strategy == "global_throttle":
                        value = float(self.config["global_keep_probability"])
                        pair = (value, value)
                    elif strategy == "static_l1":
                        pair = (
                            float(self.config["static_l1_keep_intra"]),
                            float(self.config["static_l1_keep_inter"]),
                        )
                    elif strategy == "static_cost_matched":
                        if direction not in selected_cost[condition_id]:
                            raise ValueError(
                                f"no calibrated static cost match for {condition_id}/"
                                f"{direction}"
                            )
                        pair = selected_cost[condition_id][direction]
                    else:
                        if direction not in selected_theory[condition_id]:
                            raise ValueError(
                                f"no calibrated theory candidate for {condition_id}/"
                                f"{direction}"
                            )
                        pair = selected_theory[condition_id][direction]
                    row = await self._run_one(
                        phase="evaluation",
                        condition=condition,
                        label=strategy,
                        strategy=strategy,
                        seed=seed,
                        keep_pair=pair,
                    )
                    evaluation_rows.append(row)
                    condition_rows.append(row)
                    self._checkpoint(
                        "evaluation",
                        condition_id,
                        condition_rows,
                        len(STRATEGIES) * len(self.config["evaluation_seeds"]),
                    )
        evaluation_rows.sort(
            key=lambda row: (
                str(row["condition"]), str(row["strategy"]), int(row["seed"])
            )
        )
        evaluation = self._evaluation_analysis(evaluation_rows)
        _write_jsonl(self.output / "evaluation_runs.jsonl", evaluation_rows)
        _write_json(self.output / "evaluation_statistics.json", evaluation)
        total_duration = time.monotonic() - wall_started
        summary = {
            "status": "completed",
            "experiment_id": self.config["experiment_id"],
            "config_sha256": self.config_sha256,
            "implementation_sha256": self.implementation_sha256,
            "preflight": preflight,
            "paper_native": paper_native,
            "formal_calibration_runs": len(calibration_rows),
            "formal_evaluation_runs": len(evaluation_rows),
            "calibration": calibration,
            "evaluation": evaluation,
            "network_validation": self.validation,
            "actual_runtime_seconds": total_duration,
            "llm_calls": 0,
        }
        _write_json(self.output / "summary.json", summary)
        self._write_report(summary)
        manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "completed",
                "actual_runtime_seconds": total_duration,
                "formal_calibration_runs_completed": len(calibration_rows),
                "formal_evaluation_runs_completed": len(evaluation_rows),
                "summary_sha256": sha256_file(self.output / "summary.json"),
                "implementation_sha256": self.implementation_sha256,
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
    pilot = ThresholdPilot(args.config, args.output, resume=args.resume)
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
