"""Integration tests: the semantic layer against the real manifest and database.

They need `uv run dbt build` to have run (manifest.json and data/f1.duckdb);
otherwise they are skipped. Expected values were checked against official F1
statistics.
"""

from collections.abc import Iterator

import duckdb
import pytest

from f1_sql_agent.config import DATABASE_PATH, MANIFEST_PATH, SEMANTIC_DIR
from f1_sql_agent.semantic.layer import SemanticLayer, load_semantic_layer

pytestmark = pytest.mark.skipif(
    not (MANIFEST_PATH.is_file() and DATABASE_PATH.is_file()),
    reason="needs the dbt build output (manifest.json and data/f1.duckdb)",
)

# Example id -> expected rows, verified against official F1 statistics.
EXPECTED = {
    "season_champion": [("Max Verstappen", 395.5)],
    "wins_by_name_without_accents": [("Kimi R\u00e4ikk\u00f6nen", 21)],
    "starts_with_shared_cars": [("Juan Fangio", 51)],
    "entries_versus_starts": [("Damon Hill", 122, 115)],
    "poles_before_2003": [("Ayrton Senna", 65)],
    "top_constructor_per_season": [
        (2014, "Mercedes", 16),
        (2015, "Mercedes", 16),
        (2016, "Mercedes", 19),
        (2017, "Mercedes", 12),
        (2018, "Mercedes", 11),
        (2019, "Mercedes", 15),
        (2020, "Mercedes", 13),
        (2021, "Red Bull", 11),
        (2022, "Red Bull", 17),
        (2023, "Red Bull", 21),
        (2024, "Red Bull", 9),
    ],
}
# Examples whose result has no single official value get a sanity test instead.
SANITY_CHECKED = {"pit_stop_trend"}


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    # Loading also validates every reference against the manifest.
    return load_semantic_layer(MANIFEST_PATH, SEMANTIC_DIR)


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    yield connection
    connection.close()


def test_every_metric_runs(
    layer: SemanticLayer, con: duckdb.DuckDBPyConnection
) -> None:
    for metric in layer.metrics:
        sql = f"SELECT {metric.sql} FROM marts.{metric.table}"
        assert con.sql(sql).fetchall()[0][0] is not None, metric.name


@pytest.mark.parametrize("example_id", sorted(EXPECTED))
def test_example_returns_verified_result(
    layer: SemanticLayer, con: duckdb.DuckDBPyConnection, example_id: str
) -> None:
    example = next(e for e in layer.examples if e.id == example_id)
    assert con.sql(example.sql).fetchall() == EXPECTED[example_id]


def test_pit_stop_trend_is_plausible(
    layer: SemanticLayer, con: duckdb.DuckDBPyConnection
) -> None:
    example = next(e for e in layer.examples if e.id == "pit_stop_trend")
    rows = con.sql(example.sql).fetchall()
    assert [row[0] for row in rows] == list(range(2011, 2025))
    # Time in the pit lane: tens of seconds, never minutes once red flags are out.
    assert all(15 < avg_s < 40 for _, avg_s, _ in rows)


def test_every_example_has_a_check(layer: SemanticLayer) -> None:
    # Adding an example without verifying its result must fail here.
    assert {e.id for e in layer.examples} == set(EXPECTED) | SANITY_CHECKED
