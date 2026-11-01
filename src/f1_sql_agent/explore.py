"""Facts about the dataset for the Explore page: size, tables, lineage, charts
and fun facts.

Everything is read from the built database (opened read-only) and from dbt's
manifest, so the page always describes the data the agent actually queries.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1_sql_agent.config import DATABASE_PATH
from f1_sql_agent.mcp_server import open_database
from f1_sql_agent.semantic.layer import SemanticLayer
from f1_sql_agent.semantic.manifest import EXPOSED_SCHEMA

DATASET_TITLE = "Formula 1 World Championship (1950 - 2024)"
DATASET_URL = (
    "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
)


def fetch(sql: str, path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Run one query on the database (read-only) and return its rows."""
    with open_database(path) as con:
        return con.execute(sql).df()


# --- Overview and tables --------------------------------------------------------

OVERVIEW_SQL = """
SELECT
    (SELECT count(DISTINCT season) FROM marts.dim_race) AS seasons,
    (SELECT count(*) FROM marts.dim_race) AS grands_prix,
    (SELECT count(DISTINCT circuit_id) FROM marts.dim_race) AS circuits,
    (SELECT count(*) FROM marts.dim_driver) AS drivers,
    (SELECT count(*) FROM marts.dim_constructor) AS constructors,
    (SELECT count(*) FROM marts.fct_race_results) AS race_entries,
    (SELECT count(*) FROM marts.fct_lap_times) AS laps_timed,
    (SELECT count(*) FROM marts.fct_pit_stops) AS pit_stops
"""


def table_summary(layer: SemanticLayer, path: Path = DATABASE_PATH) -> pd.DataFrame:
    """One row per table of the star schema: kind, grain, size and the
    seasons it covers (for the tables linked to a race)."""
    rows = []
    with open_database(path) as con:
        for table in layer.catalog.tables.values():
            source = f"{EXPOSED_SCHEMA}.{table.name}"
            columns = {col.name for col in table.columns}
            if table.name == "dim_race":
                seasons = f"SELECT min(season), max(season) FROM {source}"
            elif "race_id" in columns:
                seasons = (
                    f"SELECT min(r.season), max(r.season) FROM {source} "
                    f"JOIN {EXPOSED_SCHEMA}.dim_race AS r USING (race_id)"
                )
            else:
                seasons = "SELECT NULL, NULL"
            first, last = con.execute(seasons).fetchall()[0]
            count = con.execute(f"SELECT count(*) FROM {source}").fetchall()[0][0]
            rows.append(
                {
                    "table": table.name,
                    "type": "dimension" if table.name.startswith("dim_") else "fact",
                    "one row per": " + ".join(table.primary_key) or "-",
                    "rows": count,
                    "columns": len(table.columns),
                    "first season": first,
                    "last season": last,
                    "description": _first_sentence(table.description),
                }
            )
    summary = pd.DataFrame(rows)
    # Whole seasons, with a gap (not NaN) for the tables without races.
    seasons = ["first season", "last season"]
    summary[seasons] = summary[seasons].astype("Int64")
    return summary


def table_columns(layer: SemanticLayer, name: str) -> pd.DataFrame:
    """The columns of one table as the agent sees them."""
    table = layer.catalog.tables[name]
    return pd.DataFrame(
        [
            {
                "column": col.name,
                "type": col.data_type,
                "nullable": col.nullable,
                "joins to": (
                    f"{col.references.table}.{col.references.column}"
                    if col.references
                    else ""
                ),
                "description": " ".join(col.description.split()),
            }
            for col in table.columns
        ]
    )


def _first_sentence(text: str) -> str:
    head, separator, _ = " ".join(text.split()).partition(". ")
    return head + "." if separator else head


# --- Lineage --------------------------------------------------------------------

# From the raw files to the tables the agent queries, left to right.
LAYERS = ("raw", "seeds", "staging", "marts")
LAYER_TITLES = {
    "raw": "Raw CSV files (Kaggle)",
    "seeds": "Seeds (hand-written fixes)",
    "staging": "Staging (views: names and types)",
    "marts": "Marts (tables: star schema)",
}


@dataclass(frozen=True)
class Step:
    """One node of the dbt lineage: a raw file, a seed or a model."""

    id: str
    name: str
    layer: str
    kind: str  # csv file, seed, view or table
    path: str  # where it lives in the repository
    description: str
    parents: tuple[str, ...]
    tests: int  # dbt tests that check it


def load_lineage(manifest_path: Path) -> dict[str, Step]:
    """Every raw file, seed and model of the dbt project, with its parents."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tests: Counter[str] = Counter()
    for node in manifest["nodes"].values():
        if node["resource_type"] == "test":
            # Generic tests check one model; singular tests the models they read.
            attached = node.get("attached_node")
            tests.update([attached] if attached else node["depends_on"]["nodes"])

    steps = {
        uid: Step(
            id=uid,
            name=f"{source['name']}.csv",
            layer="raw",
            kind="csv file",
            path=f"data/raw/{source['name']}.csv",
            description=source["description"].strip(),
            parents=(),
            tests=tests[uid],
        )
        for uid, source in manifest["sources"].items()
    }
    for uid, node in manifest["nodes"].items():
        if node["resource_type"] not in ("model", "seed"):
            continue
        is_seed = node["resource_type"] == "seed"
        steps[uid] = Step(
            id=uid,
            name=node["name"],
            layer="seeds" if is_seed else node["schema"],
            kind="seed" if is_seed else node["config"]["materialized"],
            path="transform/" + node["original_file_path"].replace("\\", "/"),
            description=" ".join(node["description"].split()),
            parents=tuple(node.get("depends_on", {}).get("nodes", [])),
            tests=tests[uid],
        )
    return steps


def count_dbt_tests(manifest_path: Path) -> int:
    """Data tests defined in the dbt project (dbt build runs all of them)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sum(node["resource_type"] == "test" for node in manifest["nodes"].values())


def upstream(steps: dict[str, Step], uid: str) -> list[Step]:
    """A step and everything it is built from, ordered from raw to marts."""
    seen: dict[str, Step] = {}

    def visit(node_id: str) -> None:
        if node_id in seen or node_id not in steps:
            return
        for parent in steps[node_id].parents:
            visit(parent)
        seen[node_id] = steps[node_id]

    visit(uid)
    return sorted(seen.values(), key=lambda step: LAYERS.index(step.layer))


def lineage_dot(steps: dict[str, Step], focus: str | None = None) -> str:
    """Graphviz source of the lineage; with `focus`, only that step and its
    upstream. Raw files nothing reads are drawn dashed."""
    shown = {s.id: s for s in upstream(steps, focus)} if focus else steps
    used = {parent for step in steps.values() for parent in step.parents}
    lines = [
        "digraph lineage {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.12; ranksep=0.7;',
        '  node [shape=box, style="rounded,filled", fillcolor="#1a1a19",',
        '        color="#3987e5", fontcolor="#f0efec", fontname="Helvetica",',
        "        fontsize=10, height=0.3];",
        '  edge [color="#8a8981", arrowsize=0.6];',
    ]
    for index, layer in enumerate(LAYERS):
        members = [step for step in shown.values() if step.layer == layer]
        if not members:
            continue
        lines += [
            f"  subgraph cluster_{index} {{",
            f'    label="{LAYER_TITLES[layer]}"; fontcolor="#c3c2b7";',
            '    fontname="Helvetica"; fontsize=11; color="#52514e"; style=rounded;',
        ]
        for step in sorted(members, key=lambda s: s.name):
            style = ""
            if step.id == focus:
                style = ', fillcolor="#3987e5", fontcolor="#0b0b0b"'
            elif step.layer == "raw" and step.id not in used:
                style = ', style="rounded,dashed", color="#8a8981", fontcolor="#c3c2b7"'
            lines.append(f'    "{step.id}" [label="{step.name}"{style}];')
        lines.append("  }")
    for step in shown.values():
        lines += [f'  "{p}" -> "{step.id}";' for p in step.parents if p in shown]
    lines.append("}")
    return "\n".join(lines)


# --- Charts ---------------------------------------------------------------------

RACES_PER_SEASON_SQL = """
SELECT season, count(*) AS grands_prix
FROM marts.dim_race
GROUP BY season
ORDER BY season
"""

# Share of the drivers who started a race and were classified at the end.
FINISH_RATE_SQL = """
SELECT r.season,
       round(100.0 * count(*) FILTER (WHERE rr.is_classified) / count(*), 1)
           AS classified_pct
FROM marts.fct_race_results AS rr
JOIN marts.dim_race AS r USING (race_id)
WHERE rr.is_race_start
GROUP BY r.season
ORDER BY r.season
"""

TOP_DRIVERS_SQL = """
SELECT d.full_name AS driver, count(DISTINCT rr.race_id) AS wins
FROM marts.fct_race_results AS rr
JOIN marts.dim_driver AS d USING (driver_id)
WHERE rr.is_win
GROUP BY d.full_name
ORDER BY wins DESC, driver
LIMIT 10
"""

# Not "constructor": charts are drawn with Vega in JavaScript, where every
# object already has a "constructor" property, so that field comes out empty.
TOP_CONSTRUCTORS_SQL = """
SELECT c.constructor_name AS team, count(DISTINCT rr.race_id) AS wins
FROM marts.fct_race_results AS rr
JOIN marts.dim_constructor AS c USING (constructor_id)
WHERE rr.is_win
GROUP BY c.constructor_name
ORDER BY wins DESC, team
LIMIT 10
"""

# Time in the pit lane; red flag stops (a minute or more) are left out.
PIT_STOPS_SQL = """
SELECT r.season, round(avg(ps.pit_stop_duration_ms) / 1000, 2) AS avg_pit_stop_s
FROM marts.fct_pit_stops AS ps
JOIN marts.dim_race AS r USING (race_id)
WHERE ps.pit_stop_duration_ms < 60000
GROUP BY r.season
ORDER BY r.season
"""

NATIONALITIES_SQL = """
SELECT driver_nationality AS nationality, count(*) AS drivers
FROM marts.dim_driver
GROUP BY driver_nationality
ORDER BY drivers DESC, nationality
LIMIT 10
"""

CIRCUITS_SQL = """
SELECT circuit_name AS circuit, circuit_country AS country,
       circuit_latitude AS latitude, circuit_longitude AS longitude,
       count(*) AS races
FROM marts.dim_race
GROUP BY ALL
ORDER BY races DESC
"""


# --- Fun facts ------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    title: str
    value: str
    detail: str


# (title, query returning one row, value template, detail template)
FUN_FACTS = (
    (
        "Most Grand Prix starts",
        """
        SELECT d.full_name,
               count(DISTINCT rr.race_id) FILTER (WHERE rr.is_race_start) AS starts
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_driver AS d USING (driver_id)
        GROUP BY d.full_name
        ORDER BY starts DESC
        LIMIT 1
        """,
        "{starts}",
        "{full_name}",
    ),
    (
        "Youngest race winner",
        """
        SELECT d.full_name, r.race_name, r.season,
               date_sub('year', d.date_of_birth, r.race_date) AS years
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        JOIN marts.dim_driver AS d USING (driver_id)
        WHERE rr.is_win
        ORDER BY r.race_date - d.date_of_birth
        LIMIT 1
        """,
        "{years} years old",
        "{full_name}, {season} {race_name}",
    ),
    (
        "Oldest race winner",
        """
        SELECT d.full_name, r.race_name, r.season,
               date_sub('year', d.date_of_birth, r.race_date) AS years
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        JOIN marts.dim_driver AS d USING (driver_id)
        WHERE rr.is_win
        ORDER BY r.race_date - d.date_of_birth DESC
        LIMIT 1
        """,
        "{years} years old",
        "{full_name}, {season} {race_name}",
    ),
    (
        "Most wins at one Grand Prix",
        """
        SELECT d.full_name, r.race_name, count(DISTINCT rr.race_id) AS wins
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        JOIN marts.dim_driver AS d USING (driver_id)
        WHERE rr.is_win
        GROUP BY d.full_name, r.race_name
        ORDER BY wins DESC
        LIMIT 1
        """,
        "{wins} wins",
        "{full_name}, {race_name}",
    ),
    (
        "Most wins in one season",
        """
        SELECT d.full_name, r.season, count(DISTINCT rr.race_id) AS wins
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        JOIN marts.dim_driver AS d USING (driver_id)
        WHERE rr.is_win
        GROUP BY d.full_name, r.season
        ORDER BY wins DESC
        LIMIT 1
        """,
        "{wins} wins",
        "{full_name} in {season}",
    ),
    (
        "Most winners in a season",
        """
        SELECT r.season, count(DISTINCT rr.driver_id) AS winners
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        WHERE rr.is_win
        GROUP BY r.season
        ORDER BY winners DESC, r.season
        LIMIT 1
        """,
        "{winners} drivers",
        "won at least one race in {season}",
    ),
    (
        "Fastest race lap since 2004",
        """
        SELECT d.full_name, r.race_name, r.season,
               rr.fastest_lap_speed_kph AS speed
        FROM marts.fct_race_results AS rr
        JOIN marts.dim_race AS r USING (race_id)
        JOIN marts.dim_driver AS d USING (driver_id)
        WHERE rr.fastest_lap_speed_kph IS NOT NULL
        ORDER BY speed DESC
        LIMIT 1
        """,
        "{speed:.1f} km/h",
        "{full_name}, {season} {race_name}",
    ),
    (
        "Country with most races",
        """
        SELECT circuit_country, count(*) AS races
        FROM marts.dim_race
        GROUP BY circuit_country
        ORDER BY races DESC
        LIMIT 1
        """,
        "{races} races",
        "{circuit_country}",
    ),
)


def fun_facts(path: Path = DATABASE_PATH) -> list[Fact]:
    """Records computed from the data, one query each."""
    facts = []
    with open_database(path) as con:
        for title, sql, value, detail in FUN_FACTS:
            cursor = con.execute(sql)
            names = [column[0] for column in cursor.description]
            row = dict(zip(names, cursor.fetchall()[0], strict=True))
            facts.append(Fact(title, value.format(**row), detail.format(**row)))
    return facts
