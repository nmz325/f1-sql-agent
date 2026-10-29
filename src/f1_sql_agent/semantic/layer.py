"""Business knowledge of the semantic layer, and the assembled SemanticLayer.

Rules and metrics live in hand-written YAML files under semantic/. They only
reference the dbt models: every reference is checked against the catalog built
from the manifest, so the two sources of truth cannot drift apart.

Convention: identifiers in backticks inside a text (`is_race_start`,
`dim_race.is_season_finale`) are validated. Write literal values in quotes
('F', 'W'), never in backticks.
"""

import re
from collections import Counter
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from f1_sql_agent.semantic.manifest import load_catalog
from f1_sql_agent.semantic.models import Catalog, Table, _Frozen

IDENTIFIER = r"[a-z][a-z0-9_]*"
# `column` or `table.column`; SQL snippets in backticks are not matched.
_BACKTICKED = re.compile(rf"`({IDENTIFIER}(?:\.{IDENTIFIER})?)`")


class SemanticError(ValueError):
    """The business YAML is invalid or references unknown tables or columns."""


class Rule(_Frozen):
    """A business rule the agent must follow when it queries some tables."""

    id: str = Field(pattern=rf"^{IDENTIFIER}$")
    tables: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)


class Metric(_Frozen):
    """A reusable aggregate expression with an agreed meaning."""

    name: str = Field(pattern=rf"^{IDENTIFIER}$")
    table: str
    sql: str = Field(min_length=1)
    description: str = Field(min_length=1)


class _RulesFile(_Frozen):
    rules: tuple[Rule, ...]


class _MetricsFile(_Frozen):
    metrics: tuple[Metric, ...]


class SemanticLayer(_Frozen):
    """Everything the agent knows about the data model."""

    catalog: Catalog
    rules: tuple[Rule, ...]
    metrics: tuple[Metric, ...]

    def rules_for(self, table: str) -> tuple[Rule, ...]:
        """Rules that apply to `table`."""
        return tuple(rule for rule in self.rules if table in rule.tables)

    def metrics_for(self, table: str) -> tuple[Metric, ...]:
        """Metrics computed over `table`."""
        return tuple(metric for metric in self.metrics if metric.table == table)


def load_semantic_layer(manifest_path: Path, semantic_dir: Path) -> SemanticLayer:
    """Load catalog and business knowledge, and check them against each other."""
    catalog = load_catalog(manifest_path)
    rules, metrics = load_business(semantic_dir)
    errors = validate_references(catalog, rules, metrics)
    if errors:
        raise SemanticError("Invalid semantic layer:\n- " + "\n- ".join(errors))
    return SemanticLayer(catalog=catalog, rules=rules, metrics=metrics)


def load_business(semantic_dir: Path) -> tuple[tuple[Rule, ...], tuple[Metric, ...]]:
    """Parse rules.yml and metrics.yml (structure only, no reference checks)."""
    rules = _load(semantic_dir / "rules.yml", _RulesFile).rules
    metrics = _load(semantic_dir / "metrics.yml", _MetricsFile).metrics
    return rules, metrics


def validate_references(
    catalog: Catalog, rules: tuple[Rule, ...], metrics: tuple[Metric, ...]
) -> list[str]:
    """Return every problem found; an empty list means everything is consistent."""
    errors = _duplicates("rule id", [rule.id for rule in rules])
    errors += _duplicates("metric name", [metric.name for metric in metrics])
    for rule in rules:
        where = f"rule {rule.id!r}"
        errors += [
            f"{where}: unknown table {table!r}"
            for table in rule.tables
            if table not in catalog.tables
        ]
        errors += _check_identifiers(catalog, rule.text, rule.tables, where)
    for metric in metrics:
        where = f"metric {metric.name!r}"
        if metric.table not in catalog.tables:
            errors.append(f"{where}: unknown table {metric.table!r}")
            continue
        errors += _check_identifiers(
            catalog, metric.description, (metric.table,), where
        )
    return errors


def _load[T: _Frozen](path: Path, model: type[T]) -> T:
    """Parse a YAML file into `model`; unknown or missing keys are errors."""
    if not path.is_file():
        raise SemanticError(f"{path} not found.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return model.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise SemanticError(f"{path}: {exc}") from exc


def _duplicates(kind: str, values: list[str]) -> list[str]:
    counts = Counter(values)
    return [f"duplicate {kind} {value!r}" for value, n in counts.items() if n > 1]


def _check_identifiers(
    catalog: Catalog, text: str, tables: tuple[str, ...], where: str
) -> list[str]:
    """Check backticked identifiers: a table, table.column, or a column in scope."""
    scope = [catalog.tables[name] for name in tables if name in catalog.tables]
    errors: list[str] = []
    for identifier in _BACKTICKED.findall(text):
        if "." in identifier:
            table_name, column = identifier.split(".")
            known = table_name in catalog.tables and _has_column(
                catalog.tables[table_name], column
            )
        else:
            known = identifier in catalog.tables or any(
                _has_column(candidate, identifier) for candidate in scope
            )
        if not known:
            errors.append(f"{where}: unknown identifier `{identifier}`")
    return errors


def _has_column(table: Table, column: str) -> bool:
    return any(col.name == column for col in table.columns)
