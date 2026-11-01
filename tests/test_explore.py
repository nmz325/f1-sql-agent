"""Tests for the data behind the Explore page.

The lineage tests use a tiny hand-made manifest. The rest reads the real
database and is skipped without the dbt build output.
"""

import json
from pathlib import Path

import pytest

from f1_sql_agent import explore
from f1_sql_agent.config import DATABASE_PATH, MANIFEST_PATH, SEMANTIC_DIR
from f1_sql_agent.semantic.layer import load_semantic_layer

needs_build = pytest.mark.skipif(
    not (MANIFEST_PATH.is_file() and DATABASE_PATH.is_file()),
    reason="needs the dbt build output (manifest.json and data/f1.duckdb)",
)


def _node(kind: str, name: str, schema: str, parents: list[str]) -> dict:
    return {
        "resource_type": kind,
        "name": name,
        "schema": schema,
        "config": {"materialized": "table" if schema == "marts" else "view"},
        "original_file_path": f"models\\{schema}\\{name}.sql",  # Windows paths
        "description": f"{name} description.",
        "depends_on": {"nodes": parents},
    }


@pytest.fixture
def steps(tmp_path: Path) -> dict[str, explore.Step]:
    source = {"name": "races", "description": "Raw races."}
    manifest = {
        "sources": {
            "source.p.f1.races": source,
            "source.p.f1.seasons": {"name": "seasons", "description": "Unused."},
        },
        "nodes": {
            "model.p.stg_races": _node(
                "model", "stg_races", "staging", ["source.p.f1.races"]
            ),
            "model.p.dim_race": _node(
                "model", "dim_race", "marts", ["model.p.stg_races"]
            ),
            "test.p.unique_race": {
                "resource_type": "test",
                "attached_node": "model.p.dim_race",
                "depends_on": {"nodes": ["model.p.dim_race"]},
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return explore.load_lineage(path)


def test_lineage_has_files_models_parents_and_tests(
    steps: dict[str, explore.Step],
) -> None:
    race = steps["model.p.dim_race"]
    assert race.layer == "marts"
    assert race.kind == "table"
    assert race.path == "transform/models/marts/dim_race.sql"
    assert race.tests == 1
    assert steps["source.p.f1.races"].name == "races.csv"


def test_upstream_goes_from_raw_to_marts(steps: dict[str, explore.Step]) -> None:
    chain = explore.upstream(steps, "model.p.dim_race")
    assert [step.name for step in chain] == ["races.csv", "stg_races", "dim_race"]


def test_lineage_graph_marks_unused_files_and_the_focus(
    steps: dict[str, explore.Step],
) -> None:
    full = explore.lineage_dot(steps)
    assert '"source.p.f1.races" -> "model.p.stg_races";' in full
    assert 'label="seasons.csv", style="rounded,dashed"' in full
    focused = explore.lineage_dot(steps, focus="model.p.dim_race")
    assert "seasons.csv" not in focused  # not upstream of dim_race
    assert 'label="dim_race", fillcolor="#3987e5"' in focused


@needs_build
def test_table_summary_covers_every_table() -> None:
    layer = load_semantic_layer(MANIFEST_PATH, SEMANTIC_DIR)
    summary = explore.table_summary(layer).set_index("table")
    assert list(summary.index) == list(layer.catalog.tables)
    assert summary.loc["dim_race", "first season"] == 1950
    assert summary.loc["fct_pit_stops", "first season"] == 2011
    assert summary["first season"].isna().sum() == 2  # driver and constructor


@needs_build
def test_chart_queries_return_rows() -> None:
    for sql in (
        explore.RACES_PER_SEASON_SQL,
        explore.FINISH_RATE_SQL,
        explore.TOP_DRIVERS_SQL,
        explore.TOP_CONSTRUCTORS_SQL,
        explore.PIT_STOPS_SQL,
        explore.NATIONALITIES_SQL,
        explore.CIRCUITS_SQL,
    ):
        assert not explore.fetch(sql).empty


@needs_build
def test_fun_facts_match_known_records() -> None:
    facts = {fact.title: fact for fact in explore.fun_facts()}
    assert facts["Youngest race winner"].value == "18 years old"
    assert facts["Youngest race winner"].detail.startswith("Max Verstappen")
    assert facts["Most winners in a season"].detail.endswith("1982")
