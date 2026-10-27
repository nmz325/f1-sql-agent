"""Smoke tests: verify the development environment is set up correctly."""

import duckdb

import f1_sql_agent


def test_package_is_importable() -> None:
    assert f1_sql_agent.__name__ == "f1_sql_agent"


def test_duckdb_runs_a_query() -> None:
    # DuckDB 2.0: run queries through con.execute() (DB-API), which still
    # supports fetchone/fetchmany. Relations from duckdb.sql() are lazy and
    # no longer have fetchone().
    with duckdb.connect() as con:  # in-memory database
        assert con.execute("SELECT 42").fetchone() == (42,)
