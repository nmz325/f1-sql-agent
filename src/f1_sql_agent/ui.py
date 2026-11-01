"""Pieces shared by the web pages: Altair charts, the choice of a chart for an
answer's rows, and small styled elements (their CSS is app_pages/style.css).

Charts have one series, so one color: slot 1 of the validated categorical
palette for dark backgrounds. Red is the app's accent, never a data color.
"""

from dataclasses import dataclass

import altair as alt
import pandas as pd

BLUE = "#3987e5"
# Long names (drivers, tables) must not be cut on the axis.
LABEL_LIMIT = 220
# Integer columns that order the rows in time: drawn as a line.
TIME_COLUMNS = frozenset({"season", "year", "decade", "round", "lap", "lap_number"})


def line(data: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> alt.Chart:
    """One series over an integer time axis (seasons, rounds), with a tooltip
    on every point and the scale ending where the data ends."""
    base = alt.Chart(data).encode(
        x=alt.X(
            f"{x}:Q",
            title=x_title,
            axis=alt.Axis(format="d"),
            scale=alt.Scale(
                domain=[int(data[x].min()), int(data[x].max())], nice=False
            ),
        ),
        y=alt.Y(f"{y}:Q", title=y_title),
    )
    trend = base.mark_line(color=BLUE, strokeWidth=2)
    # Invisible points, larger than the line, catch the hover.
    hover = base.mark_point(size=80, opacity=0).encode(
        tooltip=[
            alt.Tooltip(f"{x}:Q", title=x_title),
            alt.Tooltip(f"{y}:Q", title=y_title),
        ]
    )
    return (trend + hover).properties(height=260)


def bars(data: pd.DataFrame, category: str, value: str, value_title: str) -> alt.Chart:
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
            x=alt.X(f"{value}:Q", title=value_title),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                alt.Tooltip(f"{value}:Q", title=value_title),
            ],
        )
        .properties(height=max(120, 28 * len(data)))
    )


@dataclass(frozen=True)
class ChartChoice:
    kind: str  # "line" or "bars"
    x: str
    y: str


def pick_chart(data: pd.DataFrame) -> ChartChoice | None:
    """A chart for an answer's rows, when one helps.

    A time column and a number give a line (points per season); a label
    column and a number give bars (wins per driver). Single rows, tables
    without numbers and results too long to read get no chart.
    """
    if not 2 <= len(data) <= 50:
        return None
    numeric = [
        column
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
        and not pd.api.types.is_bool_dtype(data[column])
    ]
    time = [column for column in numeric if column.lower() in TIME_COLUMNS]
    values = [
        column
        for column in numeric
        if column not in time and not column.lower().endswith("_id")
    ]
    labels = [
        column
        for column in data.columns
        if pd.api.types.is_string_dtype(data[column]) or data[column].dtype == object
    ]
    if not values:
        return None
    if time and len(data) >= 3 and data[time[0]].is_unique:
        return ChartChoice("line", time[0], values[0])
    if labels and data[labels[0]].is_unique:
        return ChartChoice("bars", labels[0], values[0])
    return None


def chart_for(data: pd.DataFrame) -> alt.Chart | None:
    """The chart pick_chart chooses, ready to draw, or None."""
    choice = pick_chart(data)
    if choice is None:
        return None
    title = choice.y.replace("_", " ")
    if choice.kind == "line":
        return line(data, choice.x, choice.y, choice.x.replace("_", " ").title(), title)
    return bars(data, choice.x, choice.y, title)


def overline(text: str) -> str:
    """HTML for a small red label above a section title."""
    return f'<div class="f1-overline">{text}</div>'
