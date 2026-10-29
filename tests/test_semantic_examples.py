"""Unit tests for verified examples."""

from pathlib import Path

from f1_sql_agent.semantic.layer import (
    Example,
    SemanticLayer,
    load_examples,
    validate_references,
)
from f1_sql_agent.semantic.models import Catalog, Column, Table

CATALOG = Catalog(
    tables={
        name: Table(name=name, columns=(Column(name="race_id", data_type="integer"),))
        for name in ("dim_race", "fct_results")
    }
)

JOIN_SQL = "SELECT * FROM marts.fct_results JOIN marts.dim_race USING (race_id)"


def test_example_tables_come_from_schema_qualified_names() -> None:
    example = Example(id="join", question="q", sql=JOIN_SQL)
    assert example.tables == ("dim_race", "fct_results")


def test_unknown_and_unqualified_tables_are_reported() -> None:
    examples = (
        Example(id="a", question="q", sql="SELECT 1 FROM marts.fct_missing"),
        Example(id="b", question="q", sql="SELECT 1 FROM fct_results"),
    )
    assert validate_references(CATALOG, (), (), examples) == [
        "example 'a': unknown table 'fct_missing'",
        "example 'b': SQL must use schema-qualified tables (marts.<table>)",
    ]


def test_load_examples_parses_file(tmp_path: Path) -> None:
    (tmp_path / "examples.yml").write_text(
        "examples:\n"
        "  - id: join\n"
        "    question: Which races?\n"
        "    sql: |\n"
        "      SELECT * FROM marts.fct_results\n"
        "      JOIN marts.dim_race USING (race_id)\n",
        encoding="utf-8",
    )
    (example,) = load_examples(tmp_path)
    assert example.question == "Which races?"
    assert example.tables == ("dim_race", "fct_results")


def test_layer_returns_examples_of_a_table() -> None:
    example = Example(id="join", question="q", sql=JOIN_SQL)
    other = Example(id="other", question="q", sql="SELECT 1 FROM marts.dim_race")
    layer = SemanticLayer(
        catalog=CATALOG, rules=(), metrics=(), examples=(example, other)
    )
    assert layer.examples_for("fct_results") == (example,)
    assert layer.examples_for("dim_race") == (example, other)
