# Semantic layer

What the agent knows about the data model beyond table and column names:
keys, relationships, valid values, business rules, metrics and verified
examples.

## Two sources, no duplication

| Source | Holds | Edited in |
|---|---|---|
| dbt manifest (`transform/target/manifest.json`) | Tables, columns, types, descriptions, primary and unique keys, relationships, accepted values | `transform/models/marts/*.yml` |
| This folder | Business rules, metrics and verified examples | `rules.yml`, `metrics.yml`, `examples.yml` |

Rule of thumb: if it describes what a value **is**, document it in dbt. If it
prescribes how to **calculate or filter** something, or spans several tables,
add it here.

`load_semantic_layer()` in `f1_sql_agent.semantic.layer` merges both sources
and checks every reference from this folder against the manifest, so they
cannot drift apart. The manifest is produced by any dbt command that parses
the project (`uv run dbt parse` or `uv run dbt build`, from `transform/`).

## Files

- `rules.yml`: rules the agent must follow, each attached to the tables it
  applies to (e.g. rows are entries, not starts; champions come from the
  standings of the season finale).
- `metrics.yml`: aggregate expressions with an agreed meaning (wins, poles,
  races started...).
- `examples.yml`: question/SQL pairs whose results were checked against
  official F1 statistics. They are shown to the agent as few-shot examples,
  so **never reuse these questions in the evaluation set**.

## Conventions

- Identifiers in backticks (`is_race_start`, `dim_race.is_season_finale`) are
  validated: loading fails if the table or column does not exist.
- Literal values go in single quotes (`'retired'`), never in backticks.
- Example SQL uses schema-qualified tables (`marts.<table>`); the tables of an
  example are derived from them.

## Adding a rule, metric or example

1. Edit the YAML file.
2. Run `uv run pytest`. It fails on unknown tables or columns, misspelled
   keys, metrics or examples that do not run in DuckDB, and examples without
   a verified result in `tests/test_semantic_integration.py`.

## What the agent reads

`f1_sql_agent.semantic.render` turns the layer into compact text, delivered
just in time instead of all at once:

| Function | Content | Used by (phase 3) |
|---|---|---|
| `render_catalog` | One line per table: name, grain, summary | `list_tables` tool |
| `render_table` | Columns, keys, relationships, valid values, rules and metrics of one table | `describe_table` tool |
| `render_examples` | Verified examples | System prompt |
