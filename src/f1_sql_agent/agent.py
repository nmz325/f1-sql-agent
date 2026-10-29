"""SQL agent: a LangGraph loop that answers F1 questions through the MCP server.

The graph is the simplest agent pattern, an LLM calling tools in a loop:

    START -> agent -> tools -> agent -> ... -> agent -> END
                        |
                        +-> give_up -> END   (too many failed queries)

- agent: the LLM reads the conversation and either calls tools or answers.
- tools: runs those tool calls on the MCP server and adds the results.
- A failed run_query comes back with DuckDB's error message, so the LLM can
  fix its SQL (self-correction). After MAX_QUERY_RETRIES retries it gives up.

The LLM is chosen with F1_AGENT_MODEL in .env, written as "provider:model".
Usage: uv run f1-sql-agent "Who won the 2008 drivers' championship?"
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.shared.exceptions import MCPError

from f1_sql_agent.config import PROJECT_ROOT

DEFAULT_MODEL = "groq:openai/gpt-oss-120b"
# Free tiers answer HTTP 429 when a per-minute limit is hit: the client waits
# what the API asks (retry-after header) and tries again, up to this many times.
MAX_LLM_RETRIES = 5
MAX_QUERY_RETRIES = 3  # failed run_query calls allowed after the first attempt
MAX_STEPS = 30  # LangGraph steps per question: a safety net against endless loops
EXAMPLES_URI = "f1://examples"

SYSTEM_PROMPT = """\
You are a data analyst. You answer questions about Formula 1 (seasons 1950 to \
2024) by querying a DuckDB database with the tools you have.

How to work:
1. Call list_tables, then describe_table for every table you plan to use. \
Never guess table or column names. Follow the rules that describe_table returns.
2. Before filtering on a name (driver, team, circuit), call column_values to \
find how it is spelled in the data.
3. Send one SELECT per run_query call. Write DuckDB SQL and qualify every \
table with its schema (marts.<table>).
4. If run_query returns an error, read the message, fix the SQL and try again.
5. Answer in the language of the question, briefly, using only the rows that \
run_query returned. If you used a proxy or the data is incomplete, say so. If \
the data cannot answer the question, say that instead of guessing.

Verified examples of good queries:

{examples}
"""

GIVE_UP_ANSWER = (
    f"I could not write a query that works after {MAX_QUERY_RETRIES} retries, "
    "so I cannot answer this question reliably."
)
TOO_MANY_STEPS_ANSWER = (
    "I stopped because I needed too many steps to answer this question."
)

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """What the graph carries from node to node."""

    messages: Annotated[list[AnyMessage], add_messages]  # new messages are appended
    failed_queries: int  # run_query calls that returned an error


def _log(event: str, **fields: Any) -> None:
    """One JSON line per agent step (structured logging)."""
    log.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


def build_graph(
    model: BaseChatModel,
    client: Client,
    tools: list[dict[str, Any]],
    system_prompt: str,
) -> CompiledStateGraph:
    """Wire the agent loop. The MCP client and the model come from outside,
    so tests can pass fakes."""
    llm = model.bind_tools(tools)

    async def agent(state: AgentState) -> dict[str, Any]:
        response = await llm.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        usage = response.usage_metadata or {}
        _log(
            "llm",
            tool_calls=[call["name"] for call in response.tool_calls],
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return {"messages": [response]}

    async def run_tools(state: AgentState) -> dict[str, Any]:
        results: list[ToolMessage] = []
        failed = 0
        for call in state["messages"][-1].tool_calls:
            started = time.perf_counter()
            try:
                result = await client.call_tool(call["name"], call["args"])
                text = "\n".join(b.text for b in result.content if b.type == "text")
                is_error = result.is_error
            except MCPError as exc:  # e.g. the LLM invented a tool name
                text, is_error = f"Error: {exc}", True
            _log(
                "tool",
                name=call["name"],
                args=call["args"],
                is_error=is_error,
                ms=round((time.perf_counter() - started) * 1000),
                result=text[:300],  # keep the log small
            )
            if is_error and call["name"] == "run_query":
                failed += 1
            results.append(
                ToolMessage(
                    text,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error" if is_error else "success",
                )
            )
        return {"messages": results, "failed_queries": state["failed_queries"] + failed}

    def give_up(state: AgentState) -> dict[str, Any]:
        _log("give_up", failed_queries=state["failed_queries"])
        return {"messages": [AIMessage(GIVE_UP_ANSWER)]}

    def after_agent(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AgentState) -> str:
        return "give_up" if state["failed_queries"] > MAX_QUERY_RETRIES else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", "give_up"])
    graph.add_edge("give_up", END)
    return graph.compile()


async def run_question(
    graph: CompiledStateGraph, question: str
) -> tuple[str, list[AnyMessage]]:
    """Run one question through the graph.

    Returns the final answer and every message of the run (tool calls, tool
    results, token usage), which the evaluation needs. stream_mode="values"
    yields the whole state after each step, so the messages so far are kept
    even when the run stops at the step limit.
    """
    _log("question", text=question)
    messages: list[AnyMessage] = []
    try:
        async for state in graph.astream(
            {"messages": [HumanMessage(question)], "failed_queries": 0},
            {"recursion_limit": MAX_STEPS},
            stream_mode="values",
        ):
            messages = state["messages"]
    except GraphRecursionError:
        _log("too_many_steps", limit=MAX_STEPS)
        return TOO_MANY_STEPS_ANSWER, messages
    answer = messages[-1].text
    _log("answer", text=answer)
    return answer, messages


async def ask(graph: CompiledStateGraph, question: str) -> str:
    """Run one question through the graph and return the final answer."""
    answer, _ = await run_question(graph, question)
    return answer


async def load_context(client: Client) -> tuple[list[dict[str, Any]], str]:
    """Read the tool definitions and the verified examples from the MCP server."""
    listing = await client.list_tools()
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in listing.tools
    ]
    resource = await client.read_resource(EXAMPLES_URI)
    examples = "\n".join(c.text for c in resource.contents if hasattr(c, "text"))
    return tools, SYSTEM_PROMPT.format(examples=examples)


def server_parameters() -> StdioServerParameters:
    """How to launch our MCP server as a subprocess (stdio transport)."""
    command = shutil.which("f1-mcp")  # the console script installed by uv sync
    if command is None:
        raise RuntimeError("f1-mcp not found: run the agent with `uv run`.")
    env = get_default_environment()  # the server does not get our secrets
    if "F1_DATA_DIR" in os.environ:
        env["F1_DATA_DIR"] = os.environ["F1_DATA_DIR"]
    return StdioServerParameters(command=command, env=env, cwd=PROJECT_ROOT)


def resolve_model_name(model_name: str | None = None) -> str:
    """The model to use: the argument, else F1_AGENT_MODEL, else the default."""
    return model_name or os.environ.get("F1_AGENT_MODEL", DEFAULT_MODEL)


@asynccontextmanager
async def open_agent(
    model_name: str | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """Start the MCP server, read its context and yield a ready graph.
    The server stops (and releases the database) when the block ends."""
    model_name = resolve_model_name(model_name)
    model = init_chat_model(model_name, temperature=0, max_retries=MAX_LLM_RETRIES)
    async with Client(server_parameters()) as client:
        tools, system_prompt = await load_context(client)
        _log("ready", model=model_name, tools=[t["function"]["name"] for t in tools])
        yield build_graph(model, client, tools, system_prompt)


async def _answer(question: str) -> None:
    async with open_agent() as graph:
        print(await ask(graph, question))


def main() -> None:
    """CLI: answer the question given as arguments; steps are logged to stderr."""
    load_dotenv()
    logging.basicConfig(format="%(message)s", stream=sys.stderr)
    log.setLevel(logging.INFO)  # only our steps: keep library logs (httpx...) quiet
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: uv run f1-sql-agent "your question"')
    asyncio.run(_answer(question))
