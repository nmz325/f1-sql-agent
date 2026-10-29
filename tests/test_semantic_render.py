"""Unit tests for the text the agent reads, on a tiny hand-made layer."""

import pytest

from f1_sql_agent.semantic.layer import Example, Metric, Rule, SemanticLayer
from f1_sql_agent.semantic.models import Catalog, Column, ColumnRef, Table
from f1_sql_agent.semantic.render import render_catalog, render_examples, render_table

RESULTS = Table(
    name="fct_results",
    description="One row per race entry. Includes entries that did not start.",
    primary_key=("result_id",),
    unique_keys=(("race_id", "driver_id"),),
    columns=(
        Column(
            name="result_id",
            data_type="integer",
            nullable=False,
            description="Entry id.",
        ),
        Column(
            name="race_id",
            data_type="integer",
            nullable=False,
            references=ColumnRef(table="dim_race", column="race_id"),
        ),
        Column(
            name="result_type",
            data_type="varchar",
            nullable=False,
            accepted_values=("classified", "retired"),
        ),
        Column(name="race_time_ms", data_type="bigint", description="Race time"),
    ),
)
RACES = Table(
    name="dim_race",
    description="One row per race.",
    primary_key=("race_id",),
    columns=(Column(name="race_id", data_type="integer", nullable=False),),
)
LAYER = SemanticLayer(
    catalog=Catalog(tables={"dim_race": RACES, "fct_results": RESULTS}),
    rules=(Rule(id="starts", tables=("fct_results",), text="Filter starts."),),
    metrics=(
        Metric(
            name="wins",
            table="fct_results",
            sql="count(*)\n  filter (where is_win)",
            description="Races won.",
        ),
    ),
    examples=(
        Example(
            id="champion", question="Who won?", sql="SELECT 1\nFROM marts.dim_race\n"
        ),
    ),
)


def test_catalog_has_one_line_per_table() -> None:
    assert render_catalog(LAYER) == (
        "- dim_race (one row per race_id): One row per race.\n"
        "- fct_results (one row per result_id): One row per race entry."
    )


def test_table_shows_keys_columns_rules_and_metrics() -> None:
    assert render_table(LAYER, "fct_results") == (
        "## fct_results\n\n"
        "One row per race entry. Includes entries that did not start.\n\n"
        "Grain: one row per result_id.\n\n"
        "Unique: race_id + driver_id.\n\n"
        "Columns:\n"
        "- result_id integer: Entry id.\n"
        "- race_id integer -> dim_race.race_id\n"
        "- result_type varchar: Values: 'classified', 'retired'.\n"
        "- race_time_ms bigint, nullable: Race time.\n\n"
        "Rules:\n"
        "- Filter starts.\n\n"
        "Metrics:\n"
        "- wins = count(*) filter (where is_win) -- Races won."
    )


def test_unknown_table_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Available tables: dim_race, fct_results"):
        render_table(LAYER, "fct_missing")


def test_examples_show_question_and_sql() -> None:
    assert render_examples(LAYER) == (
        "Question: Who won?\nSQL:\nSELECT 1\nFROM marts.dim_race"
    )
