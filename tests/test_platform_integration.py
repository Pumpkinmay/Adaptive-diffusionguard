from pathlib import Path

import pytest

from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy
from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform


@pytest.mark.asyncio
async def test_refresh_controls_recommendations_and_following_and_tracks_root(
    tmp_path: Path,
) -> None:
    platform = AdaptiveDiffusionPlatform(
        str(tmp_path / "platform.db"),
        user_communities={0: "a", 1: "a", 2: "b"},
        post_risk_scores={},
        policy=StaticCOSREFPolicy(0.0, 0.0, seed=3),
        random_seed=3,
        recsys_type="twitter",
        refresh_rec_post_count=10,
        following_post_count=10,
        max_rec_post_len=10,
    )
    for user_id in range(3):
        platform.db.execute(
            "INSERT INTO user (user_id, agent_id, user_name) VALUES (?, ?, ?)",
            (user_id, user_id, f"u{user_id}"),
        )
    platform.db.execute(
        "INSERT INTO follow (follower_id, followee_id) VALUES (0, 2)"
    )
    platform.db.execute(
        "INSERT INTO post (post_id, user_id, content, created_at) "
        "VALUES (1, 2, 'root', 0)"
    )
    platform.db.execute(
        "INSERT INTO post (post_id, user_id, original_post_id, created_at) "
        "VALUES (2, 1, 1, 0)"
    )
    platform.db.execute(
        "INSERT INTO post "
        "(post_id, user_id, original_post_id, quote_content, created_at) "
        "VALUES (3, 1, 2, 'nested quote', 0)"
    )
    platform.db.execute("INSERT INTO rec (user_id, post_id) VALUES (0, 3)")
    platform.db.commit()
    platform.post_risk_scores[1] = 1.0

    result = await platform.refresh(0)
    assert result["success"] is True
    assert result["posts"] == []
    rows = platform.db.execute(
        "SELECT post_id, root_post_id, author_community, source, shown "
        "FROM diffusionguard_impression ORDER BY source"
    ).fetchall()
    assert rows == [
        (1, 1, "b", "following", 0),
        (3, 1, "b", "recommendation", 0),
    ]
    # Both sources are governed; if deduplication removes the root because the
    # repost was already selected, root attribution must still be correct.
    assert all(row[-1] == 0 for row in rows)
    platform.db.close()


@pytest.mark.asyncio
async def test_refresh_formats_quote_posts_without_crashing(tmp_path: Path) -> None:
    platform = AdaptiveDiffusionPlatform(
        str(tmp_path / "quote.db"),
        user_communities={0: "a", 1: "b"},
        post_risk_scores={1: 0.0},
        policy=StaticCOSREFPolicy(1.0, 1.0, seed=7),
        random_seed=7,
        recsys_type="twitter",
        refresh_rec_post_count=10,
        following_post_count=0,
        max_rec_post_len=10,
    )
    for user_id in range(2):
        platform.db.execute(
            "INSERT INTO user (user_id, agent_id, user_name) VALUES (?, ?, ?)",
            (user_id, user_id, f"u{user_id}"),
        )
    platform.db.execute(
        "INSERT INTO post (post_id, user_id, content, created_at) "
        "VALUES (1, 1, 'root', 0)"
    )
    platform.db.execute(
        "INSERT INTO post "
        "(post_id, user_id, original_post_id, content, quote_content, "
        "created_at, num_reports) VALUES (2, 1, 1, 'root', 'context', 1, 2)"
    )
    platform.db.execute("INSERT INTO rec (user_id, post_id) VALUES (0, 2)")
    platform.db.commit()

    result = await platform.refresh(0)

    assert result["success"] is True
    assert len(result["posts"]) == 1
    assert result["posts"][0]["post_id"] == 2
    assert result["posts"][0]["num_reports"] == 2
    assert result["posts"][0]["content"].startswith("[Warning:")
    assert "Quote content: context" in result["posts"][0]["content"]
    platform.db.close()
