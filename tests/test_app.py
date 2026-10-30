"""Tests for the Streamlit page (app.py), run with Streamlit's AppTest.

The first test only draws the page. The second one asks a question end to end
(page, agent, real MCP server and database) with a scripted fake LLM, so it
needs the dbt build output and is skipped without it.
"""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from f1_sql_agent import agent
from f1_sql_agent.config import DATABASE_PATH, MANIFEST_PATH, PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


class ScriptedModel(GenericFakeChatModel):
    """Plays the LLM: returns the scripted messages in order, ignores the tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": name}])


def test_page_renders_without_asking_anything() -> None:
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.title[0].value == "F1 SQL Agent"
    assert len(at.sidebar.button) == 3  # example questions


@pytest.mark.skipif(
    not (MANIFEST_PATH.is_file() and DATABASE_PATH.is_file()),
    reason="needs the dbt build output (manifest.json and data/f1.duckdb)",
)
def test_question_shows_answer_steps_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql = "SELECT count(*) AS races FROM marts.dim_race WHERE season = 2024"
    script = [
        tool_call("list_tables", {}),
        tool_call("run_query", {"sql": sql}),
        AIMessage("There were 24 Grand Prix in 2024."),
    ]
    monkeypatch.setattr(
        agent, "init_chat_model", lambda *a, **k: ScriptedModel(messages=iter(script))
    )

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("How many races were held in 2024?").run()

    assert not at.exception
    assert "There were 24 Grand Prix in 2024." in [m.value for m in at.markdown]
    assert at.code[0].value == sql
    assert at.dataframe[0].value["races"].tolist() == [24]
    (turn,) = at.session_state.turns
    assert turn["steps"] == ["`list_tables`", "`run_query`"]
