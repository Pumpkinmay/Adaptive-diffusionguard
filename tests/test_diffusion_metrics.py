import sqlite3

from adaptive_diffusionguard.metrics.diffusion import CascadeSize


def test_cascade_size_interface() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE post (post_id INTEGER PRIMARY KEY, original_post_id INTEGER)"
    )
    connection.executemany(
        "INSERT INTO post VALUES (?, ?)", [(1, None), (2, 1), (3, 1)]
    )
    assert CascadeSize().compute(connection, 1) == 3.0
