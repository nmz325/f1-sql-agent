"""Evaluation of the agent: questions with verified answers, and how to grade them.

The set lives in eval/questions.yml. Each question has a reference ("gold")
SQL query and the rows it must return. The metric is execution accuracy, as
in the BIRD and Spider text-to-SQL benchmarks: an answer counts as correct
when the result of the agent's SQL matches the expected rows.

Commands:
    uv run f1-eval check   # the gold SQL still returns the expected rows
    uv run f1-eval run     # ask the agent every question, save its answers

`run` spends LLM tokens, so it saves one JSON line per question in
eval/results/<model>.jsonl as soon as it is answered, and skips the questions
already in that file: if the free tier runs out, run it again later and it
continues where it stopped. Delete the file to start over.

References: https://arxiv.org/abs/2305.03111 (BIRD, execution accuracy) and
https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests
"""

import argparse
import asyncio
import json
import math
import re
import sys
import time
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

import duckdb
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from f1_sql_agent.agent import (
    GIVE_UP_ANSWER,
    TOO_MANY_STEPS_ANSWER,
    open_agent,
    resolve_model_name,
    run_question,
)
from f1_sql_agent.config import DATABASE_PATH, PROJECT_ROOT, SEMANTIC_DIR
from f1_sql_agent.mcp_server import open_database
from f1_sql_agent.semantic.layer import load_examples

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.yml"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
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


# --- Running the agent --------------------------------------------------------


def results_path(model_name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """One results file per model: 'groq:openai/gpt-oss-120b' ->
    eval/results/groq-openai-gpt-oss-120b.jsonl."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return results_dir / f"{slug}.jsonl"


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read a results file (one JSON object per line); empty if it does not exist."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def outcome_of(answer: str) -> str:
    """How the run ended: answered, give_up (too many failed queries) or
    too_many_steps (the step limit stopped it)."""
    if answer == GIVE_UP_ANSWER:
        return "give_up"
    if answer == TOO_MANY_STEPS_ANSWER:
        return "too_many_steps"
    return "answered"


def summarize_run(messages: Sequence[AnyMessage]) -> dict[str, Any]:
    """What the evaluation keeps from a run: the tool calls in order, the last
    SQL that ran without error (the one that is graded) and the LLM usage."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    tool_calls: list[dict[str, Any]] = []
    sql = None
    llm_calls = input_tokens = cached_tokens = output_tokens = 0
    for message in messages:
        if not isinstance(message, AIMessage) or message.text == GIVE_UP_ANSWER:
            continue  # the give-up answer is written by the graph, not the LLM
        llm_calls += 1
        usage = message.usage_metadata or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        cached_tokens += usage.get("input_token_details", {}).get("cache_read", 0)
        for call in message.tool_calls:
            result = results.get(call["id"])
            failed = result is None or result.status == "error"
            tool_calls.append(
                {"name": call["name"], "args": call["args"], "error": failed}
            )
            if call["name"] == "run_query" and not failed:
                sql = call["args"].get("sql")
    return {
        "sql": sql,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "failed_queries": sum(
            1 for c in tool_calls if c["name"] == "run_query" and c["error"]
        ),
        "tool_calls": tool_calls,
    }


async def run_agent(questions: Sequence[Question], model_name: str, path: Path) -> None:
    """Ask the agent each question and append one JSON line per answer.

    A rate limit error stops the run (the daily quota is probably used up);
    any other error skips the question. Neither is saved, so the next run
    tries those questions again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    async with open_agent(model_name) as graph:
        for number, q in enumerate(questions, start=1):
            print(f"[{number}/{len(questions)}] {q.id}", flush=True)
            started = time.perf_counter()
            try:
                answer, messages = await run_question(graph, q.question)
            except Exception as exc:  # noqa: BLE001 - report it and keep going
                if type(exc).__name__ == "RateLimitError":
                    print("    rate limit reached: run the same command later to go on")
                    return
                print(f"    error, skipped: {exc!r}")
                continue
            record = {
                "id": q.id,
                "level": q.level,
                "model": model_name,
                "question": q.question,
                "outcome": outcome_of(answer),
                "answer": answer,
                **summarize_run(messages),
                "seconds": round(time.perf_counter() - started, 1),
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            tokens = record["input_tokens"] + record["output_tokens"]
            print(
                f"    {record['outcome']}, {'with' if record['sql'] else 'NO'} SQL, "
                f"{record['llm_calls']} LLM calls, {tokens} tokens, "
                f"{record['seconds']:.0f} s"
            )


# --- Command line ---------------------------------------------------------------


def _check_command(questions: list[Question]) -> None:
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


def _run_command(questions: list[Question], args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args.model)
    path = results_path(model_name)
    if args.only:
        unknown = set(args.only) - {q.id for q in questions}
        if unknown:
            sys.exit(f"Unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q.id in args.only]
    done = {record["id"] for record in load_results(path)}
    pending = [q for q in questions if q.id not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Model {model_name}: {len(done)} questions already in {path.name}")
    if not pending:
        print("Nothing to run.")
        return
    print(f"Running {len(pending)} question(s); each one can take a few minutes.")
    asyncio.run(run_agent(pending, model_name, path))


def main() -> None:
    """Entry point for `uv run f1-eval`."""
    load_dotenv()  # GROQ_API_KEY and F1_AGENT_MODEL, for `run`
    parser = argparse.ArgumentParser(prog="f1-eval", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="check that the gold SQL returns the expected rows"
    )
    run = commands.add_parser(
        "run", help="ask the agent the pending questions and save its answers"
    )
    run.add_argument("--model", help='"provider:model" (default: F1_AGENT_MODEL)')
    run.add_argument("--only", nargs="+", metavar="ID", help="only these question ids")
    run.add_argument("--limit", type=int, help="at most this many pending questions")
    args = parser.parse_args()

    questions = load_questions()
    if args.command == "check":
        _check_command(questions)
    else:
        _run_command(questions, args)
