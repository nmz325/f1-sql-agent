"""MCP server: read-only access to the F1 star schema for the agent.

Tools (the model decides when to call them):
- list_tables: the tables, what one row is, and a one-line summary
- describe_table: columns, keys, joins, rules and metrics of one table
- column_values: how values are actually written (accents, abbreviations)
- run_query: run one SELECT and get the rows back as CSV

Resource (the agent reads it once, for its system prompt):
- f1://examples: verified question/SQL pairs

Safety does not depend on the SQL being well-behaved: the database is opened
read-only, with file and network access off and the settings locked, so a
query can neither change data nor reach anything outside the database. On top
of that, only one SELECT is accepted per call, results are capped at MAX_ROWS
rows, and queries are cancelled after QUERY_TIMEOUT_S seconds.

Run it with `uv run f1-mcp`. It speaks MCP over stdin/stdout, so it must never
print: logs go to stderr.
References: https://py.sdk.modelcontextprotocol.io/v2/ and
https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
"""

import csv
import io
import threading
from pathlib import Path
from typing import Annotated, Any

import duckdb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from f1_sql_agent.config import (
    DATABASE_PATH,
    MANIFEST_PATH,
    MAX_ROWS,
    QUERY_TIMEOUT_S,
    SEMANTIC_DIR,
)
from f1_sql_agent.semantic.layer import SemanticLayer, load_semantic_layer
from f1_sql_agent.semantic.manifest import EXPOSED_SCHEMA
from f1_sql_agent.semantic.render import render_catalog, render_examples, render_table

EXAMPLES_URI = "f1://examples"

INSTRUCTIONS = """\
Read-only access to Formula 1 World Championship data (seasons 1950-2024) in \
DuckDB. Workflow: list_tables, then describe_table for each table you need \
(follow its rules), column_values to check how names are written, and \
run_query. Verified examples are in the resource f1://examples."""

# DuckDB settings for running SQL written by a model.
SAFE_CONFIG = {
    # No files, network, ATTACH, COPY or extensions.
    "enable_external_access": False,
    "autoload_known_extensions": False,
    "autoinstall_known_extensions": False,
    # A query cannot switch the settings above back on.
    "lock_configuration": True,
}

# Hints for MCP clients: the tools change nothing and only see this database.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def open_database(path: Path) -> duckdb.DuckDBPyConnection:
    """Open the database read-only, with the safe settings."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run dbt build` from transform/."
        )
    return duckdb.connect(str(path), read_only=True, config=SAFE_CONFIG)


def build_server(
    layer: SemanticLayer,
    con: duckdb.DuckDBPyConnection,
    *,
    max_rows: int = MAX_ROWS,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> MCPServer:
    """Create the server; the tools read from `layer` and `con`."""
    server = MCPServer("f1-sql", instructions=INSTRUCTIONS)
    tables = layer.catalog.tables

    def query(
        sql: str, parameters: dict[str, Any] | None = None
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run one SELECT with the limits. Up to max_rows + 1 rows come back,
        so the caller can tell whether there were more."""
        # Each tool call runs in its own thread, and a DuckDB connection must
        # not be shared between threads: every call gets its own cursor.
        with con.cursor() as cursor:
            _check_single_select(cursor, sql)
            # DuckDB has no timeout setting: interrupt the query from a timer.
            timer = threading.Timer(timeout_s, cursor.interrupt)
            timer.start()
            try:
                cursor.execute(sql, parameters)
                columns = [column[0] for column in cursor.description]
                # Results are streamed: the rest of the rows is never computed.
                rows = cursor.fetchmany(max_rows + 1)
            except duckdb.InterruptException:
                raise ToolError(
                    f"Query cancelled after {timeout_s:g} s. Make it cheaper: "
                    "filter early, aggregate, avoid cross joins."
                ) from None
            except duckdb.Error as exc:
                # DuckDB's message says what is wrong, so the model can fix it.
                raise ToolError(str(exc)) from exc
            finally:
                timer.cancel()
        return columns, rows

    def table_or_error(table: str) -> str:
        """Accept 'dim_race', 'DIM_RACE' or 'marts.dim_race'."""
        name = table.strip().lower().removeprefix(f"{EXPOSED_SCHEMA}.")
        if name not in tables:
            available = ", ".join(tables)
            raise ToolError(f"Unknown table {table!r}. Available tables: {available}.")
        return name

    @server.tool(annotations=READ_ONLY)
    def list_tables() -> str:
        """List the tables you can query: name, grain (what one row is) and a summary.

        Start here, then call describe_table for the tables a question needs.
        """
        return render_catalog(layer)

    @server.tool(annotations=READ_ONLY)
    def describe_table(
        table: Annotated[
            str, Field(description="Table name from list_tables, e.g. 'dim_race'.")
        ],
    ) -> str:
        """Describe one table before querying it.

        Returns its columns (type, meaning, valid values, joins to other
        tables), its keys, and the business rules and metrics that apply to
        it. Follow the rules: they say how to count starts, wins, poles and
        championships correctly.
        """
        return render_table(layer, table_or_error(table))

    @server.tool(annotations=READ_ONLY)
    def column_values(
        table: Annotated[str, Field(description="Table name, e.g. 'dim_driver'.")],
        column: Annotated[str, Field(description="Column name, e.g. 'full_name'.")],
        contains: Annotated[
            str | None,
            Field(
                description=(
                    "Optional text to search for, ignoring case and accents: "
                    "'raikkonen' finds 'Kimi R\u00e4ikk\u00f6nen'."
                )
            ),
        ] = None,
    ) -> str:
        """List up to 20 distinct values of a column, most frequent first.

        Use it to see how values are written before filtering on them: names
        of drivers, constructors, circuits and races (accents, abbreviations),
        statuses, nationalities.
        """
        name = table_or_error(table)
        columns = [col.name for col in tables[name].columns]
        if column not in columns:
            raise ToolError(
                f"{name} has no column {column!r}. Columns: {', '.join(columns)}."
            )
        # Table and column names come from the catalog, never from the model;
        # the search text is passed as a parameter.
        sql = f"""
            SELECT "{column}" AS value, count(*) AS rows_with_value
            FROM {EXPOSED_SCHEMA}."{name}"
            WHERE $contains IS NULL OR contains(
                strip_accents(lower(CAST("{column}" AS VARCHAR))),
                strip_accents(lower($contains))
            )
            GROUP BY ALL
            ORDER BY rows_with_value DESC, value
            LIMIT 20
        """
        names, rows = query(sql, {"contains": contains})
        if not rows:
            return f"No value of {name}.{column} contains {contains!r}."
        return _to_csv(names, rows)

    @server.tool(
        annotations=READ_ONLY,
        description=(
            "Run one read-only SELECT on the F1 DuckDB database and get the "
            "result as CSV.\n\n"
            f"Write tables as {EXPOSED_SCHEMA}.<table>. At most {max_rows} rows "
            "come back: aggregate in SQL instead of fetching raw rows, and use "
            "ORDER BY with LIMIT for top-N questions. Queries running over "
            f"{timeout_s:g} s are cancelled. If the query fails, the error says "
            "why: fix the SQL and try again."
        ),
    )
    def run_query(
        sql: Annotated[str, Field(description="One DuckDB SELECT statement.")],
    ) -> str:
        columns, rows = query(sql)
        if len(rows) > max_rows:
            summary = (
                f"First {max_rows} rows; there are more. Aggregate or filter in "
                "SQL, or use ORDER BY ... LIMIT."
            )
        elif not rows:
            summary = (
                "0 rows. If you filtered on a name, check how it is written "
                "with column_values."
            )
        else:
            summary = "1 row." if len(rows) == 1 else f"{len(rows)} rows."
        return summary + "\n" + _to_csv(columns, rows[:max_rows])

    @server.resource(
        EXAMPLES_URI,
        name="examples",
        title="Verified SQL examples",
        description=(
            "Question/SQL pairs whose results were checked against official "
            "F1 statistics. Meant for the agent's system prompt."
        ),
        mime_type="text/plain",
    )
    def examples() -> str:
        return render_examples(layer)

    return server


def _check_single_select(cursor: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Accept exactly one SELECT, using DuckDB's own parser (the one that runs it)."""
    try:
        statements = cursor.extract_statements(sql)
    except duckdb.Error as exc:
        raise ToolError(str(exc)) from exc  # a syntax error, with its position
    if len(statements) != 1:
        raise ToolError(f"Send exactly one SQL statement (found {len(statements)}).")
    kind = statements[0].type
    if kind != duckdb.StatementType.SELECT:
        raise ToolError(f"Only SELECT queries are allowed, not {kind.name}.")


def _to_csv(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([_cell(value) for value in row] for row in rows)
    return buffer.getvalue().rstrip("\n")


def _cell(value: Any) -> Any:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        # Points are DOUBLE: hide artifacts such as 53187.049999999996.
        return round(value, 9)
    return value


def main() -> None:
    """Entry point for `uv run f1-mcp`: serve over stdin/stdout."""
    layer = load_semantic_layer(MANIFEST_PATH, SEMANTIC_DIR)
    with open_database(DATABASE_PATH) as con:
        build_server(layer, con).run("stdio")
