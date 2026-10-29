"""Evaluation of the agent: questions with verified answers, and how to grade them.

The set lives in eval/questions.yml. Each question has a reference ("gold")
SQL query and the rows it must return. The metric is execution accuracy, as
in the BIRD and Spider text-to-SQL benchmarks: an answer counts as correct
when the result of the agent's SQL matches the expected rows.

Command:
    uv run f1-eval check   # the gold SQL still returns the expected rows

References: https://arxiv.org/abs/2305.03111 (BIRD, execution accuracy) and
https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests
"""

import argparse
import math
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

import duckdb
import yaml
from pydantic import BaseModel, ConfigDict, Field

from f1_sql_agent.config import DATABASE_PATH, PROJECT_ROOT, SEMANTIC_DIR
from f1_sql_agent.mcp_server import open_database
from f1_sql_agent.semantic.layer import load_examples

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.yml"
# Decimals may differ by rounding (30.14 vs 30.1); integers must match exactly.
REL_TOLERANCE = 0.005

Rows = Sequence[Sequence[Any]]


class Question(BaseModel):
    """One question of the evaluation set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    level: int = Field(ge=1, le=4)  # 1 lookup ... 4 window functions
    tags: tuple[str, ...] = ()
    question: str = Field(min_length=1)
    sql: str = Field(min_length=1)
    expected: tuple[tuple[Any, ...], ...]  # one tuple per row
    ordered: bool = False  # True only when the order is part of the answer


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read the evaluation set; fail on unknown fields or repeated ids."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [Question(**item) for item in data["questions"]]
    ids = [q.id for q in questions]
    repeated = sorted({qid for qid in ids if ids.count(qid) > 1})
    if repeated:
        raise ValueError(f"Repeated question ids: {repeated}")
    return questions


def _normalize(value: Any) -> Any:
    """Make values comparable: numbers as float, dates as ISO text, and text
    without accents, case or surrounding spaces."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        return float(value)
    if isinstance(value, date):  # datetime is a date too
        return value.isoformat()
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.casefold().strip()


def values_match(expected: Any, actual: Any) -> bool:
    """Equal values; decimals only need to agree within REL_TOLERANCE."""
    exp, act = _normalize(expected), _normalize(actual)
    if isinstance(exp, float) and isinstance(act, float):
        if exp.is_integer():
            return act == exp
        return math.isclose(act, exp, rel_tol=REL_TOLERANCE)
    return exp == act


def _rows_match(expected: Sequence[Any], actual: Sequence[Any]) -> bool:
    return all(values_match(e, a) for e, a in zip(expected, actual, strict=True))


def _same_rows(expected: Rows, actual: Rows, ordered: bool) -> bool:
    if ordered:
        return all(_rows_match(e, a) for e, a in zip(expected, actual, strict=True))
    remaining = list(actual)
    for exp_row in expected:  # each expected row must use up one actual row
        found = next(
            (i for i, row in enumerate(remaining) if _rows_match(exp_row, row)), None
        )
        if found is None:
            return False
        remaining.pop(found)
    return True


def results_match(expected: Rows, actual: Rows, *, ordered: bool = False) -> bool:
    """True if `actual` holds exactly the expected rows.

    Extra columns are allowed (the agent may add context, such as the race
    name next to a count), extra or missing rows are not.
    """
    if len(actual) != len(expected):
        return False
    if not expected:
        return True
    width = len(expected[0])
    # Try every choice of `width` actual columns, in every order.
    for columns in permutations(range(len(actual[0])), width):
        projected = [tuple(row[c] for c in columns) for row in actual]
        if _same_rows(expected, projected, ordered):
            return True
    return False


def overlapping_examples(
    questions: Iterable[Question], example_questions: Iterable[str]
) -> list[str]:
    """Ids of questions that are also few-shot examples: they would inflate
    the accuracy, because the agent sees their SQL in its prompt."""
    examples = {_normalize(text) for text in example_questions}
    return [q.id for q in questions if _normalize(q.question) in examples]


def check(questions: Iterable[Question], con: duckdb.DuckDBPyConnection) -> int:
    """Run every gold query, print the outcome and return the failures."""
    failures = 0
    for q in questions:
        rows = con.sql(q.sql).fetchall()
        ok = results_match(q.expected, rows, ordered=q.ordered)
        print(f"{'ok  ' if ok else 'FAIL'} level {q.level}  {q.id}")
        if not ok:
            failures += 1
            print(f"     expected: {[list(row) for row in q.expected]}")
            print(f"     got:      {[list(row) for row in rows]}")
    return failures


def main() -> None:
    """Entry point for `uv run f1-eval`."""
    parser = argparse.ArgumentParser(prog="f1-eval", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="check that the gold SQL returns the expected rows"
    )
    parser.parse_args()

    questions = load_questions()
    example_questions = [example.question for example in load_examples(SEMANTIC_DIR)]
    overlap = overlapping_examples(questions, example_questions)
    if overlap:
        sys.exit(f"These questions are also few-shot examples, remove them: {overlap}")
    with open_database(DATABASE_PATH) as con:
        failures = check(questions, con)
    passed = len(questions) - failures
    print(f"\n{passed}/{len(questions)} gold queries return the expected rows")
    if failures:
        sys.exit(1)
