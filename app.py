"""Web UI for the F1 SQL agent, with two pages picked from the sidebar:

- Ask the agent (app_pages/ask.py): questions in natural language, answered
  with the SQL behind every answer.
- Explore the data (app_pages/explore.py): the dataset, its tables, how the
  data flows from the raw files to the star schema, charts and fun facts.

Run it with `uv run streamlit run app.py`; it opens http://localhost:8501.
The F1 look is .streamlit/config.toml (red accent) plus app_pages/style.css.
Reference: https://docs.streamlit.io/develop/api-reference/navigation
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # GROQ_API_KEY and F1_AGENT_MODEL

ROOT = Path(__file__).parent
STYLE = ROOT / "app_pages" / "style.css"
# An original logo: a checkered flag on a red badge (not the Formula 1 logo).
LOGO = ROOT / "assets" / "logo.svg"
LOGO_ICON = ROOT / "assets" / "logo_icon.svg"
FAVICON = ROOT / "assets" / "favicon.png"  # the same mark, for the browser tab
REPO_URL = "https://github.com/nmz325/f1-sql-agent"
GITHUB_URL = "https://github.com/nmz325"
LINKEDIN_URL = "https://www.linkedin.com/in/nahuel-martinez-77161827b/"

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
