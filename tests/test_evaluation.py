"""Unit tests for the evaluation: question set, result comparison, run
summaries and grading (no LLM and no project database)."""

from datetime import date
from pathlib import Path

import duckdb
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from f1_sql_agent.agent import GIVE_UP_ANSWER, TOO_MANY_STEPS_ANSWER
from f1_sql_agent.config import SEMANTIC_DIR
from f1_sql_agent.evaluation import (
    Question,
    grade,
    load_questions,
    load_results,
    outcome_of,
    overlapping_examples,
    render_report,
    results_match,
    results_path,
    summarize_run,
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


def llm_turn(sql: str, call_id: str, tokens: int) -> AIMessage:
    """An LLM message that calls run_query, with its token usage."""
    return AIMessage(
        "",
        tool_calls=[{"name": "run_query", "args": {"sql": sql}, "id": call_id}],
        usage_metadata={
            "input_tokens": tokens,
            "output_tokens": 10,
            "total_tokens": tokens + 10,
        },
    )


def test_summary_keeps_the_last_query_that_worked() -> None:
    messages = [
        HumanMessage("q"),
        llm_turn("SELEC 1", "a", 100),
        ToolMessage("Parser Error", tool_call_id="a", status="error"),
        llm_turn("SELECT 1", "b", 200),
        ToolMessage("1 row.", tool_call_id="b"),
        llm_turn("SELECT 2", "c", 300),
        ToolMessage("Binder Error", tool_call_id="c", status="error"),
        AIMessage("It is 1."),
    ]
    summary = summarize_run(messages)
    assert summary["sql"] == "SELECT 1"
    assert summary["failed_queries"] == 2
    assert summary["llm_calls"] == 4
    assert summary["input_tokens"] == 600
    assert [call["error"] for call in summary["tool_calls"]] == [True, False, True]


def test_summary_without_queries_has_no_sql() -> None:
    summary = summarize_run([HumanMessage("q"), AIMessage("From memory: 7.")])
    assert summary["sql"] is None
    assert summary["llm_calls"] == 1


def test_give_up_message_is_not_counted_as_an_llm_call() -> None:
    summary = summarize_run([HumanMessage("q"), AIMessage(GIVE_UP_ANSWER)])
    assert summary["llm_calls"] == 0


def test_outcome_of_the_fixed_answers() -> None:
    assert outcome_of(GIVE_UP_ANSWER) == "give_up"
    assert outcome_of(TOO_MANY_STEPS_ANSWER) == "too_many_steps"
    assert outcome_of("Lewis Hamilton.") == "answered"


def test_results_file_is_named_after_the_model(tmp_path: Path) -> None:
    path = results_path("groq:openai/gpt-oss-120b", tmp_path)
    assert path == tmp_path / "groq-openai-gpt-oss-120b.jsonl"


def test_load_results_reads_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    assert load_results(path) == []  # no file yet
    path.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    assert [record["id"] for record in load_results(path)] == ["a", "b"]


def races_question(qid: str = "races", level: int = 1) -> Question:
    return Question(
        id=qid, level=level, question="How many?", sql="SELECT 24", expected=((24,),)
    )


def test_grade_runs_the_agent_sql_again() -> None:
    question = races_question()
    with duckdb.connect() as con:  # in memory: these queries need no tables
        assert (
            grade(question, {"sql": "SELECT 24 AS races"}, con)["status"] == "correct"
        )
        assert grade(question, {"sql": "SELECT 23"}, con)["status"] == "wrong_result"
        assert grade(question, {"sql": None}, con)["status"] == "no_sql"
        assert grade(question, {"sql": "SELEC 1"}, con)["status"] == "sql_error"


def test_report_shows_accuracy_and_failures() -> None:
    questions = [races_question("a"), races_question("b", 2), races_question("c", 2)]
    record = {
        "answer": "There were 23.",
        "sql": "SELECT 23",
        "outcome": "answered",
        "llm_calls": 4,
        "input_tokens": 1000,
        "output_tokens": 100,
        "seconds": 5.0,
        "failed_queries": 1,
    }
    records = {"a": {**record, "sql": "SELECT 24"}, "b": record}  # c was not run
    grades = {
        "a": {"status": "correct", "rows": [(24,)]},
        "b": {"status": "wrong_result", "rows": [(23,)]},
    }
    report = render_report("fake:model", questions, records, grades)
    assert "Execution accuracy: **1/2** (50%). Questions not run yet: 1." in report
    assert "| 2 | 0/1 | 0% |" in report
    assert "### b (level 2): wrong_result" in report
    assert "SELECT 23" in report
    assert "> There were 23." in report
