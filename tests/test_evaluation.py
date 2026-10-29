"""Unit tests for the evaluation set and the result comparison (no database)."""

from datetime import date
from pathlib import Path

import pytest

from f1_sql_agent.config import SEMANTIC_DIR
from f1_sql_agent.evaluation import (
    Question,
    load_questions,
    overlapping_examples,
    results_match,
)
from f1_sql_agent.semantic.layer import load_examples


def test_extra_columns_are_allowed() -> None:
    expected = [["Max Verstappen", 18]]
    actual = [("Spanish Grand Prix", "Max Verstappen", 18)]
    assert results_match(expected, actual)


def test_extra_or_missing_rows_fail() -> None:
    assert not results_match([[1]], [(1,), (2,)])
    assert not results_match([[1], [2]], [(1,)])


def test_text_ignores_accents_and_case() -> None:
    assert results_match([["Kimi Raikkonen"]], [("kimi r\u00e4ikk\u00f6nen",)])


def test_integers_must_be_exact() -> None:
    assert results_match([[67]], [(67.0,)])
    assert not results_match([[106]], [(105,)])


def test_decimals_tolerate_rounding_but_not_other_units() -> None:
    assert results_match([[30.1]], [(30.14,)])
    assert not results_match([[30.1]], [(0.301,)])  # a fraction, not a percentage


def test_dates_compare_with_iso_text() -> None:
    assert results_match([["1950-05-13"]], [(date(1950, 5, 13),)])


def test_order_only_matters_when_asked() -> None:
    expected = [["Hamilton", 105], ["Schumacher", 91]]
    actual = [("Schumacher", 91), ("Hamilton", 105)]
    assert results_match(expected, actual)
    assert not results_match(expected, actual, ordered=True)


def test_repeated_ids_are_rejected(tmp_path: Path) -> None:
    item = "  - {id: a, level: 1, question: q, sql: SELECT 1, expected: [[1]]}\n"
    path = tmp_path / "questions.yml"
    path.write_text("questions:\n" + item + item, encoding="utf-8")
    with pytest.raises(ValueError, match="Repeated question ids"):
        load_questions(path)


def test_questions_that_are_examples_are_reported() -> None:
    question = Question(
        id="a", level=1, question="Who won?", sql="SELECT 1", expected=()
    )
    assert overlapping_examples([question], ["who won?"]) == ["a"]


def test_real_question_set_is_valid() -> None:
    questions = load_questions()
    assert 20 <= len(questions) <= 30
    assert {q.level for q in questions} == {1, 2, 3, 4}
    examples = [example.question for example in load_examples(SEMANTIC_DIR)]
    assert overlapping_examples(questions, examples) == []
