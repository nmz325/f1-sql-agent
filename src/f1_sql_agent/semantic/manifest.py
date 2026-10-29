"""Build the structural part of the semantic layer from dbt's manifest.json.

Why the manifest and not the YAML files: dbt has already resolved configs,
docs and tests there, and it is the only place where relationships live
(the database has no foreign keys by design, see transform/).
Reference: https://docs.getdbt.com/reference/artifacts/manifest-json
"""

import json
from pathlib import Path
from typing import Any

from f1_sql_agent.semantic.models import Catalog, Column, ColumnRef, Table

# Artifact schemas may change in any dbt minor version: fail fast if it does.
EXPECTED_SCHEMA_VERSION = "https://schemas.getdbt.com/dbt/manifest/v12.json"
# Only the star schema is exposed to the agent.
EXPOSED_SCHEMA = "marts"

Node = dict[str, Any]
# (model unique_id, column name)
ColumnKey = tuple[str, str]


class ManifestError(RuntimeError):
    """The manifest is missing, unsupported or inconsistent."""


def load_catalog(manifest_path: Path, schema: str = EXPOSED_SCHEMA) -> Catalog:
    """Read manifest.json and return the tables of `schema` with their metadata."""
    if not manifest_path.is_file():
        raise ManifestError(
            f"{manifest_path} not found. Run `uv run dbt parse` from transform/."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    version = manifest["metadata"]["dbt_schema_version"]
    if version != EXPECTED_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema {version} "
            f"(expected {EXPECTED_SCHEMA_VERSION})."
        )

    nodes: dict[str, Node] = manifest["nodes"]
    models = {
        uid: node
        for uid, node in nodes.items()
        if node["resource_type"] == "model" and node["schema"] == schema
    }
    if not models:
        raise ManifestError(f"No models found in schema {schema!r}.")

    # Generic tests attached to exposed models carry relationships and enums.
    tests = [
        node
        for node in nodes.values()
        if node["resource_type"] == "test"
        and "test_metadata" in node
        and node.get("attached_node") in models
    ]
    references = _references(tests, models)
    accepted = _accepted_values(tests)

    tables = {
        node["name"]: _build_table(uid, node, references, accepted)
        for uid, node in models.items()
    }
    # Sorted so that everything rendered from the catalog is deterministic.
    return Catalog(tables=dict(sorted(tables.items())))


def _references(
    tests: list[Node], models: dict[str, Node]
) -> dict[ColumnKey, ColumnRef]:
    """Map each column to its target, using `relationships` tests.

    The target model is taken from the test's dependencies instead of parsing
    the Jinja string in kwargs["to"] (e.g. "ref('dim_race')").
    """
    refs: dict[ColumnKey, ColumnRef] = {}
    for test in tests:
        if test["test_metadata"]["name"] != "relationships":
            continue
        source = test["attached_node"]
        # depends_on lists both models; self-references are not supported.
        targets = [uid for uid in test["depends_on"]["nodes"] if uid != source]
        if len(targets) != 1:
            raise ManifestError(f"Cannot resolve the target of {test['unique_id']}.")
        if targets[0] not in models:
            continue  # target outside the exposed schema: useless to the agent
        key = (source, test["column_name"])
        if key in refs:
            raise ManifestError(f"{key} has more than one relationships test.")
        refs[key] = ColumnRef(
            table=models[targets[0]]["name"],
            column=test["test_metadata"]["kwargs"]["field"],
        )
    return refs


def _accepted_values(tests: list[Node]) -> dict[ColumnKey, tuple[str, ...]]:
    """Map each column to its valid values, using `accepted_values` tests."""
    values: dict[ColumnKey, tuple[str, ...]] = {}
    for test in tests:
        metadata = test["test_metadata"]
        if metadata["name"] == "accepted_values":
            key = (test["attached_node"], test["column_name"])
            values[key] = tuple(str(v) for v in metadata["kwargs"]["values"])
    return values


def _constraint_columns(node: Node, kind: str) -> list[tuple[str, ...]]:
    """Column sets of every `kind` constraint, declared on the model or a column."""
    found = [tuple(c["columns"]) for c in node["constraints"] if c["type"] == kind]
    found += [
        (name,)
        for name, col in node["columns"].items()
        if any(c["type"] == kind for c in col["constraints"])
    ]
    return found


def _build_table(
    uid: str,
    node: Node,
    references: dict[ColumnKey, ColumnRef],
    accepted: dict[ColumnKey, tuple[str, ...]],
) -> Table:
    """Assemble a Table from a model node and the facts found in its tests."""
    primary_keys = _constraint_columns(node, "primary_key")
    if len(primary_keys) > 1:
        raise ManifestError(f"{node['name']} declares more than one primary key.")
    primary_key = primary_keys[0] if primary_keys else ()

    columns = tuple(
        Column(
            name=name,
            data_type=col["data_type"],
            description=col["description"].strip(),
            # PK columns are NOT NULL even without an explicit constraint.
            nullable=name not in primary_key
            and not any(c["type"] == "not_null" for c in col["constraints"]),
            accepted_values=accepted.get((uid, name), ()),
            references=references.get((uid, name)),
        )
        for name, col in node["columns"].items()
    )
    return Table(
        name=node["name"],
        columns=columns,
        description=node["description"].strip(),
        primary_key=primary_key,
        unique_keys=tuple(_constraint_columns(node, "unique")),
    )
