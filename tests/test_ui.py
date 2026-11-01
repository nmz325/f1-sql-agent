"""Tests for the chart chosen for an answer's rows."""

import pandas as pd

from f1_sql_agent.ui import ChartChoice, chart_for, pick_chart


def test_a_number_per_season_is_a_line() -> None:
    rows = pd.DataFrame({"season": [2021, 2022, 2023], "points_margin": [8, 146, 290]})
    assert pick_chart(rows) == ChartChoice("line", "season", "points_margin")


def test_a_number_per_name_is_a_ranking() -> None:
    rows = pd.DataFrame(
        {"full_name": ["Max Verstappen", "Lando Norris"], "wins": [9, 4]}
    )
    assert pick_chart(rows) == ChartChoice("bars", "full_name", "wins")


def test_no_chart_for_one_row_or_no_numbers() -> None:
    assert (
        pick_chart(pd.DataFrame({"full_name": ["Lewis Hamilton"], "wins": [105]}))
        is None
    )
    assert (
        pick_chart(pd.DataFrame({"full_name": ["A", "B"], "team": ["X", "Y"]})) is None
    )


def test_ids_and_repeated_labels_are_not_charted() -> None:
    # Ids are not measures; repeated labels would merge into one bar.
    assert pick_chart(pd.DataFrame({"name": ["A", "B"], "driver_id": [1, 2]})) is None
    assert pick_chart(pd.DataFrame({"team": ["X", "X"], "wins": [1, 2]})) is None


def test_chart_for_draws_the_choice() -> None:
    rows = pd.DataFrame({"full_name": ["A", "B"], "wins": [2, 1]})
    assert chart_for(rows).to_dict()["mark"]["type"] == "bar"
