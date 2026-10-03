import sqlite3

from adaptive_diffusionguard.storage.impressions import (
    ImpressionRecord,
    ImpressionStore,
)


def test_normalized_impression_log_has_required_fields() -> None:
    connection = sqlite3.connect(":memory:")
    store = ImpressionStore(connection)
    store.append(
        ImpressionRecord(
            run_id="run",
            timestep=1,
            user_id=2,
            post_id=3,
            root_post_id=1,
            user_community="a",
            author_community="b",
            base_score=0.7,
            risk_score=0.9,
            omega_intra=0.8,
            omega_inter=0.2,
            keep_probability=0.28,
            shown=False,
        )
    )
    row = connection.execute(
        "SELECT run_id, root_post_id, keep_probability, shown "
        "FROM diffusionguard_impression"
    ).fetchone()
    assert row == ("run", 1, 0.28, 0)
