"""Web UI for the F1 SQL agent, with two pages picked from the sidebar:

- Ask the agent (app_pages/ask.py): questions in natural language, answered
  with the SQL behind every answer.
- Explore the data (app_pages/explore.py): the dataset, its tables, how the
  data flows from the raw files to the star schema, charts and fun facts.

Run it with `uv run streamlit run app.py`; it opens http://localhost:8501.
The F1 look is .streamlit/config.toml (red accent) plus app_pages/style.css.

It also runs on Streamlit Community Cloud: settings come from the host's
secrets, and the database is built on the first visit (see bootstrap.py).
Reference: https://docs.streamlit.io/develop/api-reference/navigation
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
try:
    import f1_sql_agent  # noqa: F401
except ImportError:  # checked out but not installed, as on some hosts
    sys.path.insert(0, str(ROOT / "src"))

from f1_sql_agent import bootstrap

STYLE = ROOT / "app_pages" / "style.css"
# An original logo: a checkered flag on a red badge (not the Formula 1 logo).
LOGO = ROOT / "assets" / "logo.svg"
LOGO_ICON = ROOT / "assets" / "logo_icon.svg"
FAVICON = ROOT / "assets" / "favicon.png"  # the same mark, for the browser tab
REPO_URL = "https://github.com/nmz325/f1-sql-agent"
GITHUB_URL = "https://github.com/nmz325"
LINKEDIN_URL = "https://www.linkedin.com/in/nahuel-martinez-77161827b/"


def load_settings() -> None:
    """Settings as environment variables, where the agent and the LLM client
    read them: the host's secrets (Streamlit Cloud), else the local .env."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file: local use
        secrets = {}
    for key, value in secrets.items():
        if isinstance(value, str | int | float):
            os.environ.setdefault(key, str(value))
    load_dotenv()  # GROQ_API_KEY, F1_AGENT_MODEL...; never overrides the above


@st.cache_resource(
    show_spinner="First visit: building the database (about a minute)..."
)
def build_database() -> bool:
    """Download the data and run dbt, once per server, if not built yet."""
    bootstrap.ensure_database()
    return True


load_settings()
ask = st.Page("app_pages/ask.py", title="Ask the agent", icon=":material/chat:")
explore = st.Page(
    "app_pages/explore.py", title="Explore the data", icon=":material/bar_chart:"
)
page = st.navigation([ask, explore])
st.set_page_config(
    page_title=f"{page.title} | F1 SQL Agent",
    page_icon=str(FAVICON),
    # Tables and charts need the room; the chat reads better narrow.
    layout="wide" if page.title == explore.title else "centered",
)
st.logo(str(LOGO), icon_image=str(LOGO_ICON), size="large", link=REPO_URL)
# Only a <style> block: it changes the look without taking room on the page.
st.html(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")
try:
    build_database()
except Exception as exc:  # noqa: BLE001 - explain it on the page
    st.error("The database could not be built. Reload the page to try again.")
    with st.expander("Details"):
        st.code(str(exc)[-3000:], language="text")
    st.stop()
page.run()

# Below each page's own sidebar content.
with st.sidebar:
    st.caption("Built by Nahuel Martinez")
    github, linkedin = st.columns(2)
    github.link_button("GitHub", GITHUB_URL, icon=":material/code:", width="stretch")
    linkedin.link_button(
        "LinkedIn", LINKEDIN_URL, icon=":material/person:", width="stretch"
    )
    st.caption("Unofficial fan project, not affiliated with Formula 1.")
