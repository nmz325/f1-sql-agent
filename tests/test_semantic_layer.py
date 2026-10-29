"""Unit tests for business rules and metrics, on a tiny hand-made catalog."""

from pathlib import Path

import pytest

from f1_sql_agent.semantic.layer import (
    Metric,
    Rule,
    SemanticError,
    SemanticLayer,
    load_business,
    validate_references,
)
from f1_sql_agent.semantic.models import Catalog, Column, Table

CATALOG = Catalog(
    tables={
        "dim_race": Table(
            name="dim_race",
            columns=(
                Column(name="race_id", data_type="integer"),
                Column(name="is_season_finale", data_type="boolean"),
            ),
        ),
        "fct_results": Table(
            name="fct_results",
            columns=(
                Column(name="race_id", data_type="integer"),
                Column(name="is_win", data_type="boolean"),
            ),
        ),
    }
)

RULES_YML = """
rules:
  - id: season_champion
    tables: [fct_results, dim_race]
    text: >-
      The champion is decided in the race where
      `dim_race.is_season_finale` is true.
"""

METRICS_YML = """
metrics:
  - name: wins
    table: fct_results
    sql: count(distinct race_id) filter (where is_win)
    description: Races won, counted once per race.
"""


def _write(directory: Path, rules: str, metrics: str) -> Path:
    (directory / "rules.yml").write_text(rules, encoding="utf-8")
    (directory / "metrics.yml").write_text(metrics, encoding="utf-8")
    return directory


def test_load_business_parses_both_files(tmp_path: Path) -> None:
    rules, metrics = load_business(_write(tmp_path, RULES_YML, METRICS_YML))
    assert [rule.id for rule in rules] == ["season_champion"]
    assert rules[0].tables == ("fct_results", "dim_race")
    assert metrics[0].sql == "count(distinct race_id) filter (where is_win)"


def test_misspelled_yaml_key_is_rejected(tmp_path: Path) -> None:
    typo = RULES_YML.replace("tables:", "tabels:")
    with pytest.raises(SemanticError, match="rules.yml"):
        load_business(_write(tmp_path, typo, METRICS_YML))


def test_valid_references_produce_no_errors() -> None:
    rule = Rule(
        id="champion",
        tables=("fct_results", "dim_race"),
        text="Join `fct_results` to `dim_race` and filter `is_season_finale`.",
    )
    metric = Metric(
        name="wins",
        table="fct_results",
        sql="count(*) filter (where is_win)",
        description="Rows where `is_win` is true.",
    )
    assert validate_references(CATALOG, (rule,), (metric,)) == []


def test_validation_reports_every_problem_at_once() -> None:
    rules = (
        Rule(id="a", tables=("fct_results", "dim_missing"), text="Use `is_win`."),
        Rule(id="a", tables=("fct_results",), text="Use `is_podium`."),
    )
    metrics = (Metric(name="m", table="fct_missing", sql="count(*)", description="x"),)
    assert validate_references(CATALOG, rules, metrics) == [
        "duplicate rule id 'a'",
        "rule 'a': unknown table 'dim_missing'",
        "rule 'a': unknown identifier `is_podium`",
        "metric 'm': unknown table 'fct_missing'",
    ]


def test_layer_returns_rules_and_metrics_of_a_table() -> None:
    rule = Rule(id="champion", tables=("fct_results", "dim_race"), text="x")
    metric = Metric(name="wins", table="fct_results", sql="count(*)", description="x")
    layer = SemanticLayer(catalog=CATALOG, rules=(rule,), metrics=(metric,))
    assert layer.rules_for("dim_race") == (rule,)
    assert layer.metrics_for("dim_race") == ()
    assert layer.metrics_for("fct_results") == (metric,)
