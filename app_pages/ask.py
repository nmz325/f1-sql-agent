"""Page "Ask the agent": ask in the browser, check the SQL behind the answer.

Every question starts the agent and its MCP server, as the CLI does, and is
answered on its own (no memory of earlier questions). Under each answer the
page shows the steps of the agent, its final query and the rows that query
returns, so the answer can be checked instead of trusted.

Reference: https://docs.streamlit.io/develop/api-reference/chat
"""

import asyncio
import time
from typing import Any

import duckdb
import streamlit as st

from f1_sql_agent.agent import open_agent, resolve_model_name, run_question
from f1_sql_agent.config import DATABASE_PATH, MAX_ROWS
from f1_sql_agent.evaluation import summarize_run
from f1_sql_agent.mcp_server import open_database

EXAMPLES = (
    "Who won the 2016 drivers' championship, and by how many points?",
    "\u00bfQu\u00e9 constructor gan\u00f3 m\u00e1s carreras en los a\u00f1os 80?",
    "How many pole positions does Lewis Hamilton have?",
)


async def _ask(question: str) -> tuple[str, list[Any]]:
    async with open_agent() as graph:
        return await run_question(graph, question)


def _step_label(call: dict[str, Any]) -> str:
    """One tool call as a short line, e.g. `column_values` dim_driver.full_name."""
    args = call["args"]
    detail = ""
    if call["name"] == "describe_tables":
        detail = ", ".join(args.get("tables", []))
        if args.get("find_names"):
            detail += f" (names: {', '.join(args['find_names'])})"
    elif call["name"] == "column_values":
        detail = f"{args.get('table')}.{args.get('column')}"
        if args.get("contains"):
            detail += f" ({args['contains']})"
    label = f"`{call['name']}` {detail}".strip()
    return label + (" - failed, then fixed" if call["error"] else "")


def _final_rows(sql: str) -> list[dict[str, Any]] | None:
    """Run the agent's final query again (read-only) to show its rows."""
    try:
        with open_database(DATABASE_PATH) as con:
            cursor = con.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS)
    except duckdb.Error:
        return None
    return [dict(zip(columns, row, strict=True)) for row in rows]


def ask_agent(question: str) -> dict[str, Any]:
    """Ask the agent and keep what the page shows: answer, steps and evidence."""
    started = time.perf_counter()
    answer, messages = asyncio.run(_ask(question))
    run = summarize_run(messages)
    return {
        "question": question,
        "answer": answer,
        "sql": run["sql"],
        "rows": _final_rows(run["sql"]) if run["sql"] else None,
        "steps": [_step_label(call) for call in run["tool_calls"]],
        "llm_calls": run["llm_calls"],
        "tokens": run["input_tokens"] + run["output_tokens"],
        "seconds": time.perf_counter() - started,
    }


def _root_cause(exc: BaseException) -> BaseException:
    """Errors inside the MCP client can arrive wrapped in exception groups."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


# The agent answers wearing a racing helmet.
AGENT_AVATAR = ":material/sports_motorsports:"
# Shown while the agent works; style.css animates the five start lights.
START_LIGHTS = """
<div class="f1-loader">
  <div class="f1-lights"><span></span><span></span><span></span><span></span><span></span></div>
  <div class="f1-loader-text">Lights out! The agent is reading the tables and
  writing SQL...</div>
</div>
"""


def show_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        st.markdown(turn["answer"])
        with st.expander("How this answer was computed"):
            st.markdown("**Agent steps**")
            steps = [f"{n}. {step}" for n, step in enumerate(turn["steps"], start=1)]
            st.markdown("\n".join(steps) or "No tool calls.")
            if turn["sql"]:
                st.markdown("**Final query and its result**")
                st.code(turn["sql"].strip(), language="sql")
                if turn["rows"] is not None:
                    st.dataframe(turn["rows"], hide_index=True)
            else:
                st.warning(
                    "The agent answered without a working query: nothing to check."
                )
            st.caption(
                f"{turn['llm_calls']} LLM calls | {turn['tokens']:,} tokens | "
                f"{turn['seconds']:.0f} s"
            )


st.title("F1 SQL Agent")
st.html('<div class="f1-stripe"></div>')
st.caption(
    "Ask about 75 years of Formula 1 (1950-2024), in English or Spanish. The "
    "agent writes and runs SQL on a DuckDB database and shows the query behind "
    "every answer."
)

with st.sidebar:
    st.subheader("Try a question")
    clicks = [st.button(example, width="stretch") for example in EXAMPLES]
    st.divider()
    st.caption(f"Model: `{resolve_model_name()}`")
    st.caption(
        "Each question is answered on its own: the agent does not remember "
        "earlier ones. Every answer uses about 6,000-12,000 tokens of the "
        "Groq free tier."
    )
    st.divider()  # before the author links that app.py adds

if "turns" not in st.session_state:
    st.session_state.turns = []
for turn in st.session_state.turns:
    show_turn(turn)

clicked = next((ex for ex, hit in zip(EXAMPLES, clicks, strict=True) if hit), None)
question = st.chat_input("Ask a question about F1 history") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        lights = st.empty()
        lights.html(START_LIGHTS)
        try:
            new_turn = ask_agent(question)
        except Exception as exc:  # noqa: BLE001 - show any failure on the page
            error = _root_cause(exc)
            if type(error).__name__ == "RateLimitError":
                st.error("The Groq free tier is used up for now. Try again later.")
            else:
                st.error(f"Something went wrong: {error!r}")
            new_turn = None
        finally:
            lights.empty()
    if new_turn is not None:
        st.session_state.turns.append(new_turn)
        st.rerun()  # draw the history again, now with the new answer
