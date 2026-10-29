"""Tests for the agent loop, with no LLM, no network and no database.

A scripted fake model plays the LLM, and a tiny in-process MCP server plays
our real one: its run_query only accepts `SELECT 1`, any other SQL fails.
"""

import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from f1_sql_agent.agent import (
    GIVE_UP_ANSWER,
    MAX_QUERY_RETRIES,
    ask,
    build_graph,
    load_context,
)

fake_server = MCPServer("fake-f1")


@fake_server.tool()
def run_query(sql: str) -> str:
    """Run one SELECT."""
    if sql != "SELECT 1":
        raise ToolError("Parser Error: syntax error")
    return "1 row\nanswer\n1"


@fake_server.resource("f1://examples")
def examples() -> str:
    """Verified examples."""
    return "Question: example?\nSQL:\nSELECT 1"


class ScriptedModel(GenericFakeChatModel):
    """Plays the LLM: returns the scripted messages in order, ignores the tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def query(sql: str) -> AIMessage:
    """An LLM turn that calls run_query."""
    return AIMessage(
        "", tool_calls=[{"name": "run_query", "args": {"sql": sql}, "id": sql}]
    )


def run(script: list[AIMessage]) -> dict[str, Any]:
    """Run one question with a scripted model and return the final graph state."""

    async def go() -> dict[str, Any]:
        async with Client(fake_server) as client:
            tools, system_prompt = await load_context(client)
            model = ScriptedModel(messages=iter(script))
            graph = build_graph(model, client, tools, system_prompt)
            return await graph.ainvoke(
                {"messages": [("user", "q")], "failed_queries": 0}
            )

    return asyncio.run(go())


def test_context_has_tools_and_examples() -> None:
    async def go() -> tuple[list[dict[str, Any]], str]:
        async with Client(fake_server) as client:
            return await load_context(client)

    tools, system_prompt = asyncio.run(go())
    assert [t["function"]["name"] for t in tools] == ["run_query"]
    assert "sql" in tools[0]["function"]["parameters"]["properties"]
    assert "Question: example?" in system_prompt


def test_answers_after_a_successful_query() -> None:
    state = run([query("SELECT 1"), AIMessage("The answer is 1.")])
    result = state["messages"][2]  # user, LLM tool call, tool result, answer
    assert result.status == "success"
    assert "1 row" in result.text
    assert state["messages"][-1].text == "The answer is 1."
    assert state["failed_queries"] == 0


def test_error_goes_back_to_the_model_and_it_retries() -> None:
    state = run([query("SELEC 1"), query("SELECT 1"), AIMessage("Fixed: 1.")])
    error = state["messages"][2]
    assert error.status == "error"
    assert "syntax error" in error.text  # DuckDB's message reaches the LLM
    assert state["messages"][-1].text == "Fixed: 1."
    assert state["failed_queries"] == 1


def test_gives_up_after_max_retries() -> None:
    bad = [query(f"SELEC {i}") for i in range(MAX_QUERY_RETRIES + 1)]
    state = run([*bad, AIMessage("never reached: the graph stops first")])
    assert state["failed_queries"] == MAX_QUERY_RETRIES + 1
    assert state["messages"][-1].text == GIVE_UP_ANSWER


def test_ask_returns_the_final_text() -> None:
    async def go() -> str:
        async with Client(fake_server) as client:
            tools, system_prompt = await load_context(client)
            model = ScriptedModel(
                messages=iter([query("SELECT 1"), AIMessage("It is 1.")])
            )
            return await ask(build_graph(model, client, tools, system_prompt), "q")

    assert asyncio.run(go()) == "It is 1."
