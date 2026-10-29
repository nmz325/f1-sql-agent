"""Tests of the MCP server through a real MCP client, connected in-process.

A tiny semantic layer and database are built for each test, so these tests do
not need the dbt build output.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import duckdb
import pytest
from mcp import Client

from f1_sql_agent.mcp_server import EXAMPLES_URI, build_server, open_database
from f1_sql_agent.semantic.layer import Example, SemanticLayer
from f1_sql_agent.semantic.models import Catalog, Column, Table

DRIVERS = Table(
    name="dim_driver",
    description="One row per driver.",
    primary_key=("driver_id",),
    columns=(
        Column(name="driver_id", data_type="integer", nullable=False),
        Column(name="full_name", data_type="varchar"),
        Column(name="points", data_type="double"),
    ),
)
LAYER = SemanticLayer(
    catalog=Catalog(tables={"dim_driver": DRIVERS}),
    rules=(),
    metrics=(),
    examples=(
        Example(id="all", question="Who?", sql="SELECT * FROM marts.dim_driver"),
    ),
)
MAX_ROWS = 2

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA marts")
        con.execute(
            "CREATE TABLE marts.dim_driver AS SELECT * FROM (VALUES "
            "(1, 'Kimi R\u00e4ikk\u00f6nen', 53187.049999999996::DOUBLE), "
            "(2, 'Juan Fangio', NULL), (3, 'Ayrton Senna', 1.0)"
            ") AS t(driver_id, full_name, points)"
        )
    with open_database(path) as con:
        server = build_server(LAYER, con, max_rows=MAX_ROWS, timeout_s=0.5)
        async with Client(server) as connected:
            yield connected


async def call(client: Client, tool: str, **arguments: object) -> tuple[bool, str]:
    """Return (is_error, text) of a tool call."""
    result = await client.call_tool(tool, arguments)
    return result.is_error, result.content[0].text


async def test_exposes_four_read_only_tools(client: Client) -> None:
    tools = (await client.list_tools()).tools
    assert [tool.name for tool in tools] == [
        "list_tables",
        "describe_table",
        "column_values",
        "run_query",
    ]
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)


async def test_describe_table_lists_the_tables_when_unknown(client: Client) -> None:
    is_error, text = await call(client, "describe_table", table="drivers")
    assert is_error
    assert "Available tables: dim_driver" in text


async def test_column_values_ignores_accents_and_case(client: Client) -> None:
    is_error, text = await call(
        client,
        "column_values",
        table="dim_driver",
        column="full_name",
        contains="RAIKK",
    )
    assert not is_error
    assert text == "value,rows_with_value\nKimi R\u00e4ikk\u00f6nen,1"


async def test_run_query_returns_csv(client: Client) -> None:
    is_error, text = await call(
        client,
        "run_query",
        sql="SELECT full_name, points FROM marts.dim_driver WHERE driver_id <= 2",
    )
    assert not is_error
    # NULL is spelled out and the floating point artifact is rounded away.
    assert text == (
        "2 rows.\nfull_name,points\nKimi R\u00e4ikk\u00f6nen,53187.05\nJuan Fangio,NULL"
    )


async def test_run_query_says_when_rows_were_cut(client: Client) -> None:
    _, text = await call(client, "run_query", sql="SELECT * FROM marts.dim_driver")
    assert text.startswith(f"First {MAX_ROWS} rows; there are more.")
    assert len(text.splitlines()) == 1 + 1 + MAX_ROWS  # summary, header, rows


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT 1; DROP TABLE marts.dim_driver", "exactly one SQL statement"),
        ("DELETE FROM marts.dim_driver", "not DELETE"),
        ("COPY marts.dim_driver TO 'stolen.csv'", "not COPY"),
        # A SELECT that passes the check is still stopped by the connection.
        ("SELECT * FROM read_csv('secret.csv')", "disabled by configuration"),
    ],
)
async def test_dangerous_queries_are_refused(
    client: Client, sql: str, message: str
) -> None:
    is_error, text = await call(client, "run_query", sql=sql)
    assert is_error
    assert message in text


async def test_run_query_returns_database_errors(client: Client) -> None:
    is_error, text = await call(
        client, "run_query", sql="SELECT driver_name FROM marts.dim_driver"
    )
    assert is_error
    # DuckDB's message suggests the right column: the model can fix the query.
    assert 'Referenced column "driver_name" not found' in text


async def test_slow_queries_are_cancelled(client: Client) -> None:
    # hash() stops DuckDB from computing the sum without reading every row.
    is_error, text = await call(
        client, "run_query", sql="SELECT sum(hash(range)) FROM range(1000000000000)"
    )
    assert is_error
    assert "cancelled after 0.5 s" in text


async def test_examples_are_a_resource(client: Client) -> None:
    content = (await client.read_resource(EXAMPLES_URI)).contents[0]
    assert content.text == "Question: Who?\nSQL:\nSELECT * FROM marts.dim_driver"
