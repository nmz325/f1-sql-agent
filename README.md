# F1 SQL Agent

An AI agent that answers natural-language questions about Formula 1 history
by writing and running SQL against a DuckDB database.

> Work in progress.

## Stack

- **DuckDB**: embedded analytical database
- **LangGraph**: agent orchestration
- **MCP**: database tools exposed through a Model Context Protocol server
- **Groq**: LLM inference (free tier)
- **uv**: Python and dependency management

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone <repo-url>
cd f1-sql-agent
uv sync
cp .env.example .env   # then add your GROQ_API_KEY
uv run pytest
```

## Web app

```bash
uv run streamlit run app.py   # http://localhost:8501
```

Two pages: **Ask the agent** (questions in English or Spanish, with the SQL
behind every answer) and **Explore the data** (the dataset, its tables, the
lineage from the raw files to the star schema, charts and fun facts).