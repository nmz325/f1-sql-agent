"""Web UI for the F1 SQL agent, with two pages picked from the sidebar:

- Ask the agent (app_pages/ask.py): questions in natural language, answered
  with the SQL behind every answer.
- Explore the data (app_pages/explore.py): the dataset, its tables, how the
  data flows from the raw files to the star schema, charts and fun facts.

Run it with `uv run streamlit run app.py`; it opens http://localhost:8501.
Reference: https://docs.streamlit.io/develop/api-reference/navigation
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # GROQ_API_KEY and F1_AGENT_MODEL

ask = st.Page("app_pages/ask.py", title="Ask the agent", icon=":material/chat:")
explore = st.Page(
    "app_pages/explore.py", title="Explore the data", icon=":material/bar_chart:"
)
page = st.navigation([ask, explore])
st.set_page_config(
    page_title=f"{page.title} | F1 SQL Agent",
    page_icon=":checkered_flag:",
    # Tables and charts need the room; the chat reads better narrow.
    layout="wide" if page.title == explore.title else "centered",
)
page.run()
