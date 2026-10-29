"""F1 SQL agent: answers Formula 1 questions with SQL over DuckDB."""


def main() -> None:
    """Console entry point (uv run f1-sql-agent): run the agent CLI."""
    # Imported here so that importing the package stays light (the MCP
    # server imports f1_sql_agent.config and does not need LangGraph).
    from f1_sql_agent.agent import main as run_agent

    run_agent()
