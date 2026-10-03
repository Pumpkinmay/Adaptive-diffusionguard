"""Small deterministic SBM smoke simulation over the OASIS platform."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import tomllib
from oasis.social_platform.typing import RecsysType

from adaptive_diffusionguard.governance.controller import RuleBasedController
from adaptive_diffusionguard.governance.cosref import (
    GlobalThrottlePolicy,
    NoInterventionPolicy,
    StaticCOSREFPolicy,
)
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sbm_edges(
    node_count: int,
    community_count: int,
    p_intra: float,
    p_inter: float,
    rng: random.Random,
) -> tuple[dict[int, str], list[tuple[int, int]]]:
    communities = {
        user_id: f"community-{user_id % community_count}"
        for user_id in range(node_count)
    }
    edges: list[tuple[int, int]] = []
    for follower in range(node_count):
        for followee in range(node_count):
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


def make_policy(
    baseline: str, config: dict[str, Any], seed: int
) -> tuple[StaticCOSREFPolicy, RuleBasedController | None]:
    policy_cfg = config["policy"]
    controller = None
    if baseline == "no_intervention":
        policy: StaticCOSREFPolicy = NoInterventionPolicy(seed)
    elif baseline == "global_throttle":
        policy = GlobalThrottlePolicy(policy_cfg["global_keep_probability"], seed)
    elif baseline in {"static_cosref", "dynamic_cosref"}:
        policy = StaticCOSREFPolicy(
            policy_cfg["omega_intra"], policy_cfg["omega_inter"], seed
        )
        if baseline == "dynamic_cosref":
            controller = RuleBasedController(**config["controller"])
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    return policy, controller


async def run(config: dict[str, Any], baseline: str, output: Path) -> dict[str, Any]:
    simulation = config["simulation"]
    node_count = int(simulation["nodes"])
    if not 50 <= node_count <= 200:
        raise ValueError("the smoke SBM must use 50 to 200 nodes")
    community_count = int(simulation["communities"])
    if community_count != 4:
        raise ValueError("the first-version smoke configuration requires 4 communities")
    seed = int(simulation["seed"])
    # OASIS' built-in random recommender uses the module-level RNG.
    random.seed(seed)
    rng = random.Random(seed)
    communities, edges = sbm_edges(
        node_count,
        community_count,
        float(simulation["p_intra"]),
        float(simulation["p_inter"]),
        rng,
    )
    policy, controller = make_policy(baseline, config, seed)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / f"{baseline}.db"
    if db_path.exists():
        db_path.unlink()
    platform = AdaptiveDiffusionPlatform(
        str(db_path),
        user_communities=communities,
        post_risk_scores={},
        policy=policy,
        controller=controller,
        random_seed=seed,
        run_id=f"sbm-{seed}-{baseline}",
        controller_window=int(simulation["controller_window"]),
        intervention_budget=float(config["controller"]["intervention_budget"]),
        recsys_type=RecsysType.RANDOM,
        refresh_rec_post_count=int(simulation["recommendation_count"]),
        following_post_count=int(simulation["following_count"]),
        max_rec_post_len=int(simulation["recommendation_buffer"]),
    )
    for user_id in range(node_count):
        platform.db.execute(
            """
            INSERT INTO user
            (user_id, agent_id, user_name, name, bio, created_at,
             num_followings, num_followers)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                user_id,
                user_id,
                f"user-{user_id}",
                f"User {user_id}",
                f"synthetic member of {communities[user_id]}",
                0,
            ),
        )
    for follower, followee in edges:
        platform.db.execute(
            "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
            (follower, followee, 0),
        )
    platform.db.commit()

    risk_posts: list[int] = []
    benign_posts: list[int] = []
    posts_per_kind = int(simulation["seed_posts_per_kind"])
    for index in range(posts_per_kind):
        risk_author = index % node_count
        result = await platform.create_post(
            risk_author, f"synthetic risky claim {index}"
        )
        post_id = int(result["post_id"])
        platform.post_risk_scores[post_id] = float(simulation["risk_score"])
        risk_posts.append(post_id)
        benign_author = (index + node_count // 2) % node_count
        result = await platform.create_post(
            benign_author, f"synthetic benign information {index}"
        )
        post_id = int(result["post_id"])
        platform.post_risk_scores[post_id] = 0.0
        benign_posts.append(post_id)

    refreshes = reposts = reports = 0
    for timestep in range(1, int(simulation["timesteps"]) + 1):
        platform.sandbox_clock.time_step = timestep
        await platform.update_rec_table()
        active_users = rng.sample(
            list(range(node_count)), int(simulation["active_users_per_step"])
        )
        for user_id in active_users:
            feed = await platform.refresh(user_id)
            refreshes += 1
            for post in feed.get("posts", []):
                root_id, _ = platform._root_post(int(post["post_id"]))
                risk = platform.post_risk_scores.get(root_id, 0.0)
                if risk and rng.random() < float(simulation["report_probability"]):
                    await platform.report_post(user_id, (int(post["post_id"]), "risk"))
                    reports += 1
                elif rng.random() < (
                    float(simulation["risk_repost_probability"])
                    if risk
                    else float(simulation["benign_repost_probability"])
                ):
                    result = await platform.repost(user_id, int(post["post_id"]))
                    reposts += int(bool(result.get("success")))

    impression_count, shown_count = platform.db.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(shown), 0)
        FROM diffusionguard_impression WHERE run_id = ?
        """,
        (platform.run_id,),
    ).fetchone()
    summary = {
        "baseline": baseline,
        "seed": seed,
        "nodes": node_count,
        "communities": community_count,
        "edges": len(edges),
        "timesteps": int(simulation["timesteps"]),
        "refreshes": refreshes,
        "candidate_impressions": int(impression_count),
        "shown_impressions": int(shown_count),
        "successful_reposts": reposts,
        "reports": reports,
        "intervention_budget": platform.intervention_budget,
        "intervention_cost": platform._cumulative_intervention_cost,
        "final_omega_intra": platform.policy.omega_intra,
        "final_omega_inter": platform.policy.omega_inter,
        "llm_agents_used": 0,
        "note": "CPU smoke run used scripted agents; no LLM inference.",
        "database": str(db_path),
    }
    (output / f"{baseline}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    platform.db_cursor.close()
    platform.db.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=(
            "no_intervention",
            "global_throttle",
            "static_cosref",
            "dynamic_cosref",
            "all",
        ),
        default="static_cosref",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baselines = (
        ["no_intervention", "global_throttle", "static_cosref", "dynamic_cosref"]
        if args.baseline == "all"
        else [args.baseline]
    )
    summaries = [
        asyncio.run(run(config, baseline, args.output)) for baseline in baselines
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
