"""Unit tests for the manifest loader, on a tiny hand-made manifest."""

import json
from pathlib import Path
from typing import Any

import pytest

from f1_sql_agent.semantic.manifest import (
    EXPECTED_SCHEMA_VERSION,
    ManifestError,
    load_catalog,
)
from f1_sql_agent.semantic.models import ColumnRef

PROJECT = "f1_sql_agent"


def _column(data_type: str, description: str = "", *constraints: str) -> dict[str, Any]:
    return {
        "data_type": data_type,
        "description": description,
        "constraints": [{"type": c} for c in constraints],
    }


def _model(
    name: str,
    columns: dict[str, Any],
    schema: str = "marts",
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "unique_id": f"model.{PROJECT}.{name}",
        "resource_type": "model",
        "name": name,
        "schema": schema,
        "description": f"{name} description",
        "constraints": constraints or [],
        "columns": columns,
    }


def _test(
    kind: str, model: str, column: str, depends_on: list[str], **kwargs: Any
) -> dict[str, Any]:
    return {
        "unique_id": f"test.{PROJECT}.{kind}_{model}_{column}",
        "resource_type": "test",
        "attached_node": f"model.{PROJECT}.{model}",
        "column_name": column,
        "depends_on": {"nodes": [f"model.{PROJECT}.{m}" for m in depends_on]},
        "test_metadata": {"name": kind, "kwargs": {"column_name": column, **kwargs}},
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    nodes = [
        _model("dim_race", {"race_id": _column("integer", "Race id.", "primary_key")}),
        _model(
            "fct_results",
            {
                "result_id": _column("integer", "Entry id.", "primary_key"),
                "race_id": _column("integer", "", "not_null"),
                "result_type": _column("varchar", "Outcome."),
            },
        ),
        _model(
            "fct_laps",
            {"race_id": _column("integer"), "lap_number": _column("integer")},
            constraints=[{"type": "primary_key", "columns": ["race_id", "lap_number"]}],
        ),
        _model("stg_races", {"race_id": _column("integer")}, schema="staging"),
        _test(
            "relationships",
            "fct_results",
            "race_id",
            ["dim_race", "fct_results"],
            to="ref('dim_race')",
            field="race_id",
        ),
        _test(
            "accepted_values",
            "fct_results",
            "result_type",
            ["fct_results"],
            values=["classified", "retired"],
        ),
    ]
    manifest = {
        "metadata": {"dbt_schema_version": EXPECTED_SCHEMA_VERSION},
        "nodes": {node["unique_id"]: node for node in nodes},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_only_exposed_schema_is_loaded(manifest_path: Path) -> None:
    tables = load_catalog(manifest_path).tables
    assert set(tables) == {"dim_race", "fct_results", "fct_laps"}


def test_primary_keys_from_column_and_model_constraints(manifest_path: Path) -> None:
    tables = load_catalog(manifest_path).tables
    assert tables["fct_results"].primary_key == ("result_id",)
    assert tables["fct_laps"].primary_key == ("race_id", "lap_number")
    # PK columns are never nullable, even without an explicit not_null.
    assert not tables["fct_laps"].column("lap_number").nullable


def test_relationships_and_accepted_values_attach_to_columns(
    manifest_path: Path,
) -> None:
    results = load_catalog(manifest_path).tables["fct_results"]
    race_id = results.column("race_id")
    assert race_id.references == ColumnRef(table="dim_race", column="race_id")
    assert not race_id.nullable
    result_type = results.column("result_type")
    assert result_type.accepted_values == ("classified", "retired")
    assert result_type.nullable


def test_unsupported_schema_version_fails_fast(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["metadata"]["dbt_schema_version"] = (
        "https://schemas.getdbt.com/dbt/manifest/v13.json"
    )
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ManifestError, match="Unsupported manifest schema"):
        load_catalog(manifest_path)


def test_missing_manifest_explains_how_to_create_it(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="dbt parse"):
        load_catalog(tmp_path / "manifest.json")
