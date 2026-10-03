"""Deterministic, LLM-free OASIS calibration and baseline comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oasis.social_platform.typing import RecsysType

from adaptive_diffusionguard.governance.controller import RuleBasedController
from adaptive_diffusionguard.governance.cosref import (
    NoInterventionPolicy,
    PolicyDecision,
    StaticCOSREFPolicy,
)
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

from .allocation import allocate_cosref_control, paper_exponential_cost, project_l1_cost
from .calibration import (
    CalibrationObservation,
    select_calibrated_keep_probabilities,
)
from .mixing import compute_mixing_statistics

BASELINES = (
    "no_intervention",
    "global_throttle",
    "static_cosref",
    "dynamic_cosref",
    "theory_informed_cosref",
)


class UniformThrottlePolicy(StaticCOSREFPolicy):
    """Project baseline that throttles risky and benign candidates equally."""

    def __init__(self, keep_probability: float, seed: int) -> None:
        super().__init__(keep_probability, keep_probability, seed)

    def keep_probability(
        self,
        user_community: str,
        author_community: str,
        risk_score: float,
    ) -> float:
        del user_community, author_community, risk_score
        return self.omega_intra

    def decide(
        self,
        user_community: str,
        author_community: str,
        risk_score: float,
    ) -> PolicyDecision:
        probability = self.keep_probability(
            user_community, author_community, risk_score
        )
        return PolicyDecision(probability, self._rng.random() < probability)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    condition: str
    baseline: str
    seed: int
    measured_mu: float
    omega_intra_initial: float
    omega_inter_initial: float
    omega_intra_final: float
    omega_inter_final: float
    candidate_impressions: int
    shown_impressions: int
    suppressed_impressions: int
    high_risk_candidates: int
    high_risk_exposures: int
    intra_high_risk_exposures: int
    inter_high_risk_exposures: int
    benign_candidates: int
    benign_exposure_loss: float
    successful_reposts: int
    successful_risk_reposts: int
    risk_cascade_size: int
    community_coverage: float
    realized_intervention_cost: float
    nominal_project_cost_initial: float
    nominal_paper_cost_initial: float
    reports: int
    llm_calls: int
    database: str
    provenance: str = "project_adaptation_oasis_experiment"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_directed_sbm(
    *,
    nodes: int,
    p_intra: float,
    p_inter: float,
    seed: int,
) -> tuple[dict[int, str], list[tuple[int, int]]]:
    if nodes < 4 or nodes % 2:
        raise ValueError("nodes must be an even integer of at least four")
    rng = random.Random(seed)
    communities = {
        node: ("community-a" if node < nodes // 2 else "community-b")
        for node in range(nodes)
    }
    edges = []
    for follower in range(nodes):
        for followee in range(nodes):
            if follower == followee:
                continue
            probability = (
                p_intra
                if communities[follower] == communities[followee]
                else p_inter
            )
            if rng.random() < probability:
                edges.append((follower, followee))
    return communities, edges


def _policy_for(
    baseline: str,
    config: Mapping[str, Any],
    seed: int,
    theory_omega: tuple[float, float] | None,
) -> tuple[StaticCOSREFPolicy, RuleBasedController | None]:
    budget = float(config["project_budget"])
    if baseline == "no_intervention":
        return NoInterventionPolicy(seed), None
    if baseline == "global_throttle":
        return UniformThrottlePolicy(float(config["global_keep_probability"]), seed), None
    if baseline == "static_cosref":
        return (
            StaticCOSREFPolicy(
                float(config["static_omega_intra"]),
                float(config["static_omega_inter"]),
                seed,
            ),
            None,
        )
    if baseline == "dynamic_cosref":
        controller_config = config["dynamic_controller"]
        return (
            StaticCOSREFPolicy(1.0, 1.0, seed),
            RuleBasedController(
                interval=int(controller_config["interval"]),
                step_size=float(controller_config["step_size"]),
                risk_target=float(controller_config["risk_target"]),
                cross_community_target=float(
                    controller_config["cross_community_target"]
                ),
                report_target=float(controller_config["report_target"]),
                benign_loss_limit=float(controller_config["benign_loss_limit"]),
                intervention_budget=budget,
            ),
        )
    if baseline == "theory_informed_cosref" and theory_omega is not None:
        return StaticCOSREFPolicy(*theory_omega, seed=seed), None
    if baseline == "calibration" and theory_omega is not None:
        return StaticCOSREFPolicy(*theory_omega, seed=seed), None
    raise ValueError(f"unsupported baseline: {baseline}")


def _has_prior_action(
    platform: AdaptiveDiffusionPlatform,
    user_id: int,
    post_id: int,
    action: str,
) -> bool:
    if action == "report":
        row = platform.db.execute(
            "SELECT 1 FROM report WHERE user_id = ? AND post_id = ? LIMIT 1",
            (user_id, post_id),
        ).fetchone()
        return row is not None
    root_id, _ = platform._root_post(post_id)
    rows = platform.db.execute(
        "SELECT post_id FROM post WHERE user_id = ? AND original_post_id IS NOT NULL",
        (user_id,),
    ).fetchall()
    return any(platform._root_post(int(row[0]))[0] == root_id for row in rows)


async def _scripted_action(
    platform: AdaptiveDiffusionPlatform,
    user_id: int,
    timestep: int,
    seed: int,
    feed: list[dict[str, Any]],
) -> None:
    risky: list[int] = []
    benign: list[int] = []
    for post in feed:
        post_id = int(post["post_id"])
        root_id, _ = platform._root_post(post_id)
        if platform.post_risk_scores[root_id] > 0:
            risky.append(post_id)
        else:
            benign.append(post_id)
    for post_id in risky:
        prefer_repost = (user_id * 7 + timestep * 3 + seed) % 4 == 0
        action_order = ("repost", "report") if prefer_repost else ("report", "repost")
        for action in action_order:
            if _has_prior_action(platform, user_id, post_id, action):
                continue
            if action == "repost":
                result = await platform.repost(user_id, post_id)
            else:
                result = await platform.report_post(
                    user_id, (post_id, "synthetic deterministic risk flag")
                )
            if result.get("success"):
                return
    if benign and (user_id + timestep + seed) % 5 == 0:
        post_id = benign[0]
        if not _has_prior_action(platform, user_id, post_id, "repost"):
            await platform.repost(user_id, post_id)


async def run_scenario(
    *,
    condition: Mapping[str, Any],
    config: Mapping[str, Any],
    baseline: str,
    seed: int,
    output_dir: Path,
    theory_omega: tuple[float, float] | None = None,
) -> ScenarioResult:
    simulation = config["simulation"]
    communities, edges = generate_directed_sbm(
        nodes=int(simulation["nodes"]),
        p_intra=float(condition["p_intra"]),
        p_inter=float(condition["p_inter"]),
        seed=seed,
    )
    mixing = compute_mixing_statistics(edges, communities)
    if mixing.mu is None:
        raise ValueError("generated scenario unexpectedly has no edges")
    policy, controller = _policy_for(baseline, config, seed, theory_omega)
    initial_omega = (policy.omega_intra, policy.omega_inter)
    output_dir.mkdir(parents=True, exist_ok=False)
    db_path = output_dir / "simulation.db"
    random.seed(seed)
    platform = AdaptiveDiffusionPlatform(
        str(db_path),
        user_communities=communities,
        post_risk_scores={},
        policy=policy,
        controller=controller,
        random_seed=seed,
        run_id=f"cosref-{condition['id']}-{baseline}-{seed}",
        controller_window=int(simulation["controller_window"]),
        intervention_budget=float(config["project_budget"]),
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
                f"user-{user_id}",
                f"Synthetic User {user_id}",
                f"synthetic profile in {community}",
            ),
        )
    platform.db.executemany(
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, 0)",
        edges,
    )
    platform.db.commit()

    nodes = int(simulation["nodes"])
    seed_specs = (
        (0, "Synthetic high-risk claim A", float(simulation["risk_score"])),
        (nodes // 2, "Synthetic high-risk claim B", float(simulation["risk_score"])),
        (1, "Synthetic verified benign update A", 0.0),
        (nodes // 2 + 1, "Synthetic verified benign update B", 0.0),
    )
    risk_roots: set[int] = set()
    for author, content, risk in seed_specs:
        result = await platform.create_post(author, content)
        post_id = int(result["post_id"])
        platform.post_risk_scores[post_id] = risk
        if risk > 0:
            risk_roots.add(post_id)

    schedule_rng = random.Random(seed + 10_000)
    all_users = list(range(nodes))
    for timestep in range(1, int(simulation["timesteps"]) + 1):
        platform.sandbox_clock.time_step = timestep
        await platform.update_rec_table()
        active_users = schedule_rng.sample(
            all_users, int(simulation["active_users_per_step"])
        )
        for user_id in active_users:
            refresh = await platform.refresh(user_id)
            await _scripted_action(
                platform,
                user_id,
                timestep,
                seed,
                list(refresh.get("posts", [])),
            )

    impression_rows = platform.db.execute(
        """
        SELECT user_id, user_community, author_community, risk_score,
               keep_probability, shown
        FROM diffusionguard_impression WHERE run_id = ?
        """,
        (platform.run_id,),
    ).fetchall()
    risky_rows = [row for row in impression_rows if float(row[3]) > 0]
    benign_rows = [row for row in impression_rows if float(row[3]) == 0]
    risky_shown = [row for row in risky_rows if bool(row[5])]
    all_posts = platform.db.execute(
        "SELECT post_id, original_post_id, quote_content FROM post"
    ).fetchall()
    risk_cascade_ids: set[int] = set()
    risk_reposts = 0
    total_reposts = 0
    for post_id, parent_id, quote_content in all_posts:
        root_id, _ = platform._root_post(int(post_id))
        if root_id in risk_roots:
            risk_cascade_ids.add(int(post_id))
        if parent_id is not None and quote_content is None:
            total_reposts += 1
            risk_reposts += int(root_id in risk_roots)
    reports = int(
        platform.db.execute("SELECT COUNT(*) FROM report").fetchone()[0]
    )
    result = ScenarioResult(
        condition=str(condition["id"]),
        baseline=baseline,
        seed=seed,
        measured_mu=float(mixing.mu),
        omega_intra_initial=initial_omega[0],
        omega_inter_initial=initial_omega[1],
        omega_intra_final=platform.policy.omega_intra,
        omega_inter_final=platform.policy.omega_inter,
        candidate_impressions=len(impression_rows),
        shown_impressions=sum(bool(row[5]) for row in impression_rows),
        suppressed_impressions=sum(not bool(row[5]) for row in impression_rows),
        high_risk_candidates=len(risky_rows),
        high_risk_exposures=len(risky_shown),
        intra_high_risk_exposures=sum(row[1] == row[2] for row in risky_shown),
        inter_high_risk_exposures=sum(row[1] != row[2] for row in risky_shown),
        benign_candidates=len(benign_rows),
        benign_exposure_loss=(
            sum(not bool(row[5]) for row in benign_rows) / len(benign_rows)
            if benign_rows
            else 0.0
        ),
        successful_reposts=total_reposts,
        successful_risk_reposts=risk_reposts,
        risk_cascade_size=len(risk_cascade_ids),
        community_coverage=(
            len({row[1] for row in risky_shown}) / len(set(communities.values()))
            if risky_shown
            else 0.0
        ),
        realized_intervention_cost=sum(1.0 - float(row[4]) for row in impression_rows),
        nominal_project_cost_initial=project_l1_cost(*initial_omega),
        nominal_paper_cost_initial=paper_exponential_cost(*initial_omega),
        reports=reports,
        llm_calls=0,
        database=str(db_path),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    platform.db_cursor.close()
    platform.db.close()
    return result


def _aggregate(results: list[ScenarioResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[(result.condition, result.baseline)].append(result)
    aggregate = []
    numeric_fields = (
        "measured_mu",
        "candidate_impressions",
        "shown_impressions",
        "suppressed_impressions",
        "high_risk_candidates",
        "high_risk_exposures",
        "intra_high_risk_exposures",
        "inter_high_risk_exposures",
        "benign_exposure_loss",
        "successful_reposts",
        "successful_risk_reposts",
        "risk_cascade_size",
        "community_coverage",
        "realized_intervention_cost",
        "omega_intra_final",
        "omega_inter_final",
    )
    for (condition, baseline), rows in sorted(grouped.items()):
        record: dict[str, object] = {
            "condition": condition,
            "baseline": baseline,
            "runs": len(rows),
            "seeds": sorted(row.seed for row in rows),
            "llm_calls": 0,
        }
        for field in numeric_fields:
            record[f"mean_{field}"] = sum(
                float(getattr(row, field)) for row in rows
            ) / len(rows)
        aggregate.append(record)
    return aggregate


async def run_bridge(config: Mapping[str, Any], output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    calibration_root = output / "oasis-calibration"
    comparison_root = output / "baseline-comparison"
    calibration_root.mkdir()
    comparison_root.mkdir()
    grid = [tuple(map(float, pair)) for pair in config["omega_grid"]]
    selections: dict[str, tuple[float, float]] = {}
    allocation_records: dict[str, object] = {}
    calibration_records: list[dict[str, object]] = []

    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        seed = int(config["calibration_seeds"][0])
        communities, edges = generate_directed_sbm(
            nodes=int(config["simulation"]["nodes"]),
            p_intra=float(condition["p_intra"]),
            p_inter=float(condition["p_inter"]),
            seed=seed,
        )
        mixing = compute_mixing_statistics(edges, communities)
        if mixing.mu is None:
            raise ValueError("calibration network has no edges")
        allocation = allocate_cosref_control(
            mixing.mu,
            float(config["project_budget"]),
            grid,
            tolerance=float(config["mu_transition_tolerance"]),
        )
        observations: list[CalibrationObservation] = []
        candidate_counter: Counter[tuple[float, float]] = Counter()
        for candidate in allocation.candidates:
            if not candidate.direction_consistent:
                continue
            omega = (candidate.omega_intra, candidate.omega_inter)
            candidate_counter[omega] += 1
            result = await run_scenario(
                condition=condition,
                config=config,
                baseline="calibration",
                seed=seed,
                output_dir=(
                    calibration_root
                    / condition_id
                    / f"omega-{omega[0]:.1f}-{omega[1]:.1f}"
                ),
                theory_omega=omega,
            )
            observation = CalibrationObservation(
                omega_intra=omega[0],
                omega_inter=omega[1],
                seed=seed,
                high_risk_exposures=result.high_risk_exposures,
                intra_high_risk_exposures=result.intra_high_risk_exposures,
                inter_high_risk_exposures=result.inter_high_risk_exposures,
                successful_risk_reposts=result.successful_risk_reposts,
                cascade_size=result.risk_cascade_size,
                community_coverage=result.community_coverage,
                benign_exposure_loss=result.benign_exposure_loss,
                intervention_cost=result.realized_intervention_cost,
            )
            observations.append(observation)
            calibration_records.append(
                {"condition": condition_id, **observation.as_dict()}
            )
        if any(count != 1 for count in candidate_counter.values()):
            raise RuntimeError("duplicate calibration candidate")
        selection = select_calibrated_keep_probabilities(observations, allocation)
        selections[condition_id] = (
            selection.omega_intra,
            selection.omega_inter,
        )
        allocation_records[condition_id] = {
            "mixing": mixing.as_dict(),
            "allocation": allocation.as_dict(),
            "selection": selection.as_dict(),
        }

    results: list[ScenarioResult] = []
    for condition in config["conditions"]:
        condition_id = str(condition["id"])
        for seed in map(int, config["evaluation_seeds"]):
            for baseline in BASELINES:
                result = await run_scenario(
                    condition=condition,
                    config=config,
                    baseline=baseline,
                    seed=seed,
                    output_dir=comparison_root / condition_id / baseline / str(seed),
                    theory_omega=selections[condition_id],
                )
                results.append(result)

    aggregate = _aggregate(results)
    summary = {
        "experiment": "minimal_cosref_theory_bridge",
        "provenance": "project_adaptation",
        "paper_omega_equals_keep_probability": False,
        "llm_calls": 0,
        "project_budget": float(config["project_budget"]),
        "conditions": [str(item["id"]) for item in config["conditions"]],
        "baselines": list(BASELINES),
        "calibration_selections": allocation_records,
        "calibration_observations": calibration_records,
        "runs": [result.as_dict() for result in results],
        "aggregate": aggregate,
        "pairing_note": (
            "Networks, seed posts, active-user schedules, labels, and seeds are paired. "
            "After an exposure is suppressed, downstream post state diverges, so later "
            "candidate events are not guaranteed to be event-by-event paired."
        ),
    }
    (output / "oasis_bridge_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_bridge(load_config(args.config), args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "llm_calls": summary["llm_calls"],
                "runs": len(summary["runs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
