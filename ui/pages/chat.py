from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.chat_history import render_chat_history
from components.chat_input import get_user_input
from services.documents import from_entry_dict, report_display_name
from services.rag import answer_for_entry
from state.session_state import init_session_state


st.set_page_config(
    page_title="Report Assistant — Chat",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_css() -> None:
    css_path = Path("ui/assets/styles.css")
    if css_path.is_file():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _go_to_start() -> None:
    if hasattr(st, "switch_page"):
        st.switch_page("pages/landing.py")
    else:
        st.session_state["ui_phase"] = "landing"
        st.rerun()


def main() -> None:
    init_session_state()
    _inject_css()

    selected_entry = st.session_state.get("selected_entry")
    if not selected_entry:
        st.error("Select a report before entering chat.")
        if st.button("Go to Start"):
            _go_to_start()
        return

    entry = from_entry_dict(selected_entry)
    st.markdown(
        f"""
        <div class="ra-context">
            <div class="ra-context-title">{report_display_name(entry)}</div>
            <div class="ra-context-subtitle">You are analyzing this document.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    messages = st.session_state["chat_messages"]
    render_chat_history(messages)

    user_input = get_user_input()
    if user_input:
        messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                response = answer_for_entry(user_input, entry)
            except Exception as exc:
                response = f"Something went wrong while retrieving an answer: {exc}"
        messages.append({"role": "assistant", "content": response})
        st.rerun()


main()
