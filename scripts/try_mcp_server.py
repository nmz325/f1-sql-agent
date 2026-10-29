"""Smoke test: start the MCP server and use each tool once, with the real data.

This is how the agent will connect in phase 4: it launches `uv run f1-mcp` as
a subprocess and talks to it through stdin/stdout.
Usage: uv run python scripts/try_mcp_server.py
"""

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

SERVER = StdioServerParameters(command="uv", args=["run", "--quiet", "f1-mcp"])

TOP_WINNERS = """
SELECT d.full_name,
       count(DISTINCT rr.race_id) FILTER (WHERE rr.is_win) AS wins
FROM marts.fct_race_results AS rr
JOIN marts.dim_driver AS d USING (driver_id)
GROUP BY d.full_name
ORDER BY wins DESC
LIMIT 5
"""

# (what we want to see, tool, arguments)
CALLS = [
    ("The tables the agent can use", "list_tables", {}),
    (
        "Finding a name typed without accents",
        "column_values",
        {"table": "dim_driver", "column": "full_name", "contains": "raikkonen"},
    ),
    ("A real question: top 5 winners", "run_query", {"sql": TOP_WINNERS}),
    (
        "A mistake: the error tells the model how to fix it",
        "run_query",
        {"sql": "SELECT driver_name FROM marts.dim_driver"},
    ),
    (
        "Something dangerous: refused",
        "run_query",
        {"sql": "DROP TABLE marts.dim_driver"},
    ),
]


async def main() -> None:
    async with Client(SERVER) as client:
        tools = await client.list_tools()
        print("Tools:", ", ".join(tool.name for tool in tools.tools))
        for title, tool, arguments in CALLS:
            result = await client.call_tool(tool, arguments)
            status = "ERROR" if result.is_error else "OK"
            print(f"\n=== {title} ({tool}: {status})")
            print(result.content[0].text)


if __name__ == "__main__":
    anyio.run(main)
