"""Page "Explore the data": the dataset behind the agent, in five sections.

Overview, tables, data lineage, charts and fun facts. Everything is read from
the built DuckDB database and from dbt's manifest (see f1_sql_agent.explore),
so the page describes exactly the data the agent queries.
"""

import altair as alt
import pandas as pd
import streamlit as st

from f1_sql_agent import explore
from f1_sql_agent.config import DATABASE_PATH, MANIFEST_PATH, SEMANTIC_DIR
from f1_sql_agent.semantic.layer import SemanticLayer, load_semantic_layer

# One series per chart, so one color: slot 1 of the validated palette (dark).
BLUE = "#3987e5"


@st.cache_resource
def semantic_layer() -> SemanticLayer:
    return load_semantic_layer(MANIFEST_PATH, SEMANTIC_DIR)


@st.cache_data
def query(sql: str) -> pd.DataFrame:
    return explore.fetch(sql)


@st.cache_data
def table_summary() -> pd.DataFrame:
    return explore.table_summary(semantic_layer())


@st.cache_data
def lineage() -> dict[str, explore.Step]:
    return explore.load_lineage(MANIFEST_PATH)


@st.cache_data
def fun_facts() -> list[explore.Fact]:
    return explore.fun_facts()


# Long names (drivers, tables) must not be cut on the axis.
LABEL_LIMIT = 220


def season_axis(first: int, last: int) -> alt.X:
    """Seasons as plain years, with the scale ending where the data ends."""
    return alt.X(
        "season:Q",
        title="Season",
        axis=alt.Axis(format="d"),
        scale=alt.Scale(domain=[first, last], nice=False),
    )


def line_chart(data: pd.DataFrame, x: str, y: str, y_title: str) -> alt.LayerChart:
    """A single series over the seasons, with a tooltip on every point."""
    base = alt.Chart(data).encode(
        x=season_axis(int(data[x].min()), int(data[x].max())),
        y=alt.Y(f"{y}:Q", title=y_title),
    )
    line = base.mark_line(color=BLUE, strokeWidth=2)
    # Invisible points, larger than the line, catch the hover.
    hover = base.mark_point(size=80, opacity=0).encode(
        tooltip=[
            alt.Tooltip(f"{x}:Q", title="Season"),
            alt.Tooltip(f"{y}:Q", title=y_title),
        ]
    )
    return (line + hover).properties(height=260)


def bar_chart(data: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    """A horizontal ranking, largest first."""
    return (
        alt.Chart(data)
        .mark_bar(color=BLUE, size=16, cornerRadiusEnd=4)
        .encode(
            y=alt.Y(
                f"{category}:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=LABEL_LIMIT),
            ),
            x=alt.X(f"{value}:Q", title=title),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                alt.Tooltip(f"{value}:Q", title=title),
            ],
        )
        .properties(height=300)
    )


def coverage_chart(summary: pd.DataFrame) -> alt.Chart:
    """Which seasons each table linked to races covers."""
    covered = summary.dropna(subset=["first season"]).assign(
        end=lambda df: df["last season"] + 1
    )
    return (
        alt.Chart(covered)
        .mark_bar(color=BLUE, size=14, cornerRadius=4)
        .encode(
            y=alt.Y(
                "table:N",
                sort=alt.EncodingSortField("first season"),
                title=None,
                axis=alt.Axis(labelLimit=LABEL_LIMIT),
            ),
            x=alt.X(
                "first season:Q",
                title="Season",
                axis=alt.Axis(format="d"),
                scale=alt.Scale(domain=[1950, 2025], nice=False),
            ),
            x2="end:Q",
            tooltip=["table", "first season", "last season", "rows"],
        )
        .properties(height=260)
    )


st.title("Explore the data")
st.html('<div class="f1-stripe"></div>')
st.caption(
    "The data behind the agent: where it comes from, how it is transformed and "
    "what it shows."
)

if not (DATABASE_PATH.is_file() and MANIFEST_PATH.is_file()):
    st.error(
        "The database is not built yet. Run `uv run f1-download`, then "
        "`uv run dbt build` from the transform/ folder."
    )
    st.stop()

overview_tab, tables_tab, lineage_tab, charts_tab, facts_tab = st.tabs(
    ["Overview", "Tables", "Data lineage", "Charts", "Fun facts"]
)

# --- Overview -------------------------------------------------------------------
with overview_tab:
    st.subheader("The dataset")
    st.markdown(
        f"**{explore.DATASET_TITLE}**, published on Kaggle by Rohan Rao and "
        "compiled from the Ergast Developer API. It covers every World "
        "Championship race from 1950 to 2024. The project pins version 24 of "
        "the dataset, so every answer and test is reproducible."
    )
    st.link_button("Open the dataset on Kaggle", explore.DATASET_URL)

    counts = query(explore.OVERVIEW_SQL).iloc[0]
    tiles = [
        ("Seasons", "seasons"),
        ("Grands Prix", "grands_prix"),
        ("Circuits", "circuits"),
        ("Drivers", "drivers"),
        ("Constructors", "constructors"),
        ("Race entries", "race_entries"),
        ("Laps timed", "laps_timed"),
        ("Pit stops", "pit_stops"),
    ]
    for row in (tiles[:4], tiles[4:]):
        for column, (label, key) in zip(st.columns(4), row, strict=True):
            column.metric(label, f"{counts[key]:,}", border=True)

    st.subheader("Coverage by season")
    st.caption(
        "Results and standings go back to 1950, but the detailed tables start "
        "later: qualifying in 1994 (complete from 2003), lap times in 1996, pit "
        "stops in 2011 and sprints in 2021. Outside those seasons the data is "
        "missing, not zero."
    )
    st.altair_chart(coverage_chart(table_summary()), width="stretch")

# --- Tables ---------------------------------------------------------------------
with tables_tab:
    st.subheader("The star schema")
    st.markdown(
        "Ten tables in the `marts` schema: three **dimensions** (who, which "
        "team, which race) and seven **facts** (what happened). Facts join "
        "dimensions through their ids."
    )
    summary = table_summary()
    # One readable column; dimensions are not tied to seasons.
    seasons = [
        "-" if pd.isna(first) else f"{first}-{last}"
        for first, last in zip(
            summary["first season"], summary["last season"], strict=True
        )
    ]
    st.dataframe(
        summary.drop(columns=["first season", "last season"]).assign(seasons=seasons),
        hide_index=True,
        width="stretch",
        column_order=["table", "type", "one row per", "rows", "columns", "seasons"],
        column_config={"rows": st.column_config.NumberColumn(format="localized")},
    )

    layer = semantic_layer()
    name = st.selectbox("Look inside a table", list(layer.catalog.tables))
    table = layer.catalog.tables[name]
    st.markdown(" ".join(table.description.split()))
    columns_col, rules_col = st.columns([3, 2])
    with columns_col:
        st.markdown("**Columns**")
        st.dataframe(
            explore.table_columns(layer, name), hide_index=True, width="stretch"
        )
    with rules_col:
        st.markdown("**Rules the agent follows for this table**")
        rules = layer.rules_for(name)
        st.markdown(
            "\n".join(f"- {' '.join(rule.text.split())}" for rule in rules)
            or "No specific rules."
        )
    st.markdown("**Sample rows**")
    # The name comes from the catalog, never from free text.
    st.dataframe(query(f"SELECT * FROM marts.{name} LIMIT 5"), hide_index=True)

# --- Data lineage ---------------------------------------------------------------
with lineage_tab:
    st.subheader("From the raw files to the agent")
    st.markdown(
        "Kaggle CSV files &rarr; `uv run f1-download` &rarr; **data/raw** &rarr; "
        "dbt **staging** &rarr; dbt **marts** &rarr; **semantic layer** &rarr; "
        "**MCP server** &rarr; **agent**"
    )
    steps = lineage()
    layers = [step.layer for step in steps.values()]
    tiles = [
        ("Raw CSV files", layers.count("raw")),
        ("Seeds", layers.count("seeds")),
        ("Staging views", layers.count("staging")),
        ("Mart tables", layers.count("marts")),
        ("dbt data tests", explore.count_dbt_tests(MANIFEST_PATH)),
    ]
    for column, (label, value) in zip(st.columns(5), tiles, strict=True):
        column.metric(label, value, border=True)

    st.markdown(
        "- **Raw**: the CSV files exactly as downloaded, read as text.\n"
        "- **Staging**: one view per file that renames columns, sets types and "
        "turns lap times like `1:27.452` into milliseconds.\n"
        "- **Seeds**: small hand-written tables, such as what the result codes "
        "mean and a fix for a race whose times were stored wrongly.\n"
        "- **Marts**: the star schema the agent queries, with flags such as "
        "`is_win` and `is_race_start`, enforced column types and tests on keys "
        "and relationships."
    )
    st.graphviz_chart(explore.lineage_dot(steps), width="stretch")
    st.caption(
        "Built from dbt's manifest. Dashed boxes are raw files that no table uses."
    )

    st.subheader("Trace one table")
    marts = sorted(step.name for step in steps.values() if step.layer == "marts")
    target = st.selectbox("Mart table", marts, index=marts.index("fct_race_results"))
    target_id = next(s.id for s in steps.values() if s.name == target)
    st.graphviz_chart(explore.lineage_dot(steps, focus=target_id), width="stretch")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "layer": step.layer,
                    "step": step.name,
                    "kind": step.kind,
                    "file": step.path,
                    "tests": step.tests,
                    "description": step.description,
                }
                for step in explore.upstream(steps, target_id)
            ]
        ),
        hide_index=True,
        width="stretch",
    )

# --- Charts ---------------------------------------------------------------------
with charts_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("**Grands Prix per season**")
        st.altair_chart(
            line_chart(
                query(explore.RACES_PER_SEASON_SQL),
                "season",
                "grands_prix",
                "Grands Prix",
            ),
            width="stretch",
        )
        st.caption("From 7 races in 1950 to 24 in 2024.")
    with right:
        st.markdown("**Cars that reach the finish**")
        st.altair_chart(
            line_chart(
                query(explore.FINISH_RATE_SQL),
                "season",
                "classified_pct",
                "% of starters classified",
            ),
            width="stretch",
        )
        st.caption(
            "Share of starters classified at the end of the race: reliability "
            "grew from about half the field to almost everyone."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Drivers with the most wins**")
        st.altair_chart(
            bar_chart(
                query(explore.TOP_DRIVERS_SQL), "driver", "wins", "Grand Prix wins"
            ),
            width="stretch",
        )
    with right:
        st.markdown("**Constructors with the most wins**")
        st.altair_chart(
            bar_chart(
                query(explore.TOP_CONSTRUCTORS_SQL),
                "team",
                "wins",
                "Grand Prix wins",
            ),
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Average time in the pit lane**")
        st.altair_chart(
            line_chart(
                query(explore.PIT_STOPS_SQL), "season", "avg_pit_stop_s", "Seconds"
            ),
            width="stretch",
        )
        st.caption(
            "From pit lane entry to exit, not only the stop itself. Red flag "
            "stops are left out."
        )
    with right:
        st.markdown("**Nationalities with the most drivers**")
        st.altair_chart(
            bar_chart(
                query(explore.NATIONALITIES_SQL), "nationality", "drivers", "Drivers"
            ),
            width="stretch",
        )
        st.caption(
            "Many Americans come from the Indianapolis 500, part of the "
            "championship from 1950 to 1960."
        )

    st.markdown("**Circuits**")
    circuits = query(explore.CIRCUITS_SQL)
    map_col, list_col = st.columns([2, 1])
    with map_col:
        # Circle radius in meters, growing with the races each circuit hosted.
        sized = circuits.assign(radius=20_000 + circuits["races"] * 2_500)
        st.map(
            sized, latitude="latitude", longitude="longitude", size="radius", color=BLUE
        )
    with list_col:
        st.dataframe(
            circuits[["circuit", "country", "races"]].head(10),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"{len(circuits)} circuits have hosted a Grand Prix.")

# --- Fun facts ------------------------------------------------------------------
with facts_tab:
    st.subheader("Records hidden in the data")
    facts = fun_facts()
    for start in range(0, len(facts), 4):
        for column, fact in zip(st.columns(4), facts[start : start + 4], strict=False):
            with column.container(border=True, height=170):  # equal cards
                st.metric(fact.title, fact.value)
                st.caption(fact.detail)
    st.caption("Computed from the data with one SQL query each.")
