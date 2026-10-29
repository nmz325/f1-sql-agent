"""Render the semantic layer as compact text for the agent.

The agent never receives the whole layer at once (just-in-time context): it
gets the table list first and asks for the detail of the tables it needs.
These functions produce exactly what the MCP tools will return in phase 3.
"""

from f1_sql_agent.semantic.layer import SemanticLayer
from f1_sql_agent.semantic.models import Column, Table


def render_catalog(layer: SemanticLayer) -> str:
    """One line per table: name, grain and the first sentence of its description."""
    return "\n".join(
        f"- {table.name} ({_grain(table)}): {_first_sentence(table.description)}"
        for table in layer.catalog.tables.values()
    )


def render_table(layer: SemanticLayer, name: str) -> str:
    """Everything the agent needs to query one table."""
    table = layer.catalog.tables.get(name)
    if table is None:
        # The agent reads this message and can retry with a valid name.
        available = ", ".join(layer.catalog.tables)
        raise ValueError(f"Unknown table {name!r}. Available tables: {available}.")

    parts = [
        f"## {table.name}",
        _sentence(table.description),
        f"Grain: {_grain(table)}.",
    ]
    if table.unique_keys:
        keys = "; ".join(" + ".join(key) for key in table.unique_keys)
        parts.append(f"Unique: {keys}.")
    parts.append("Columns:\n" + "\n".join(_column(col) for col in table.columns))

    rules = layer.rules_for(name)
    if rules:
        parts.append("Rules:\n" + "\n".join(f"- {_clean(r.text)}" for r in rules))
    metrics = layer.metrics_for(name)
    if metrics:
        parts.append(
            "Metrics:\n"
            + "\n".join(
                f"- {m.name} = {_clean(m.sql)} -- {_clean(m.description)}"
                for m in metrics
            )
        )
    return "\n\n".join(parts)


def render_examples(layer: SemanticLayer) -> str:
    """Verified question/SQL pairs, to show the agent how queries should look."""
    return "\n\n".join(
        f"Question: {example.question}\nSQL:\n{example.sql.strip()}"
        for example in layer.examples
    )


def _grain(table: Table) -> str:
    if not table.primary_key:
        return "no primary key"
    return "one row per " + " + ".join(table.primary_key)


def _column(col: Column) -> str:
    """Format: `- name type[, nullable][ -> table.column][: description Values: ...]`."""
    text = f"- {col.name} {col.data_type}"
    if col.nullable:
        text += ", nullable"
    if col.references:
        text += f" -> {col.references.table}.{col.references.column}"
    details = [_sentence(col.description)] if col.description else []
    if col.accepted_values:
        values = ", ".join(f"'{value}'" for value in col.accepted_values)
        details.append(f"Values: {values}.")
    if details:
        text += ": " + " ".join(details)
    return text


def _clean(text: str) -> str:
    """Collapse newlines and repeated spaces into single spaces."""
    return " ".join(text.split())


def _sentence(text: str) -> str:
    """Clean text that always ends with a period."""
    text = _clean(text)
    return text if not text or text.endswith((".", "!", "?")) else text + "."


def _first_sentence(text: str) -> str:
    head, separator, _ = _clean(text).partition(". ")
    return head + "." if separator else _sentence(head)
