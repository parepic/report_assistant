from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.components.report_picker_amar import render_report_picker_single_select
from services.documents import load_report_entries, to_entry_dict
from state.session_state import init_session_state


st.set_page_config(
    page_title="Report Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    css_path = Path("ui/assets/styles.css")
    if css_path.is_file():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _go_to_chat_page() -> None:
    if hasattr(st, "switch_page"):
        st.switch_page("pages/chat.py")
    else:
        st.session_state["ui_phase"] = "chat"
        st.rerun()


def main() -> None:
    init_session_state()
    _inject_css()

    st.markdown(
        """
        <div class="ra-landing">
            <div class="ra-hero">Boost Your Financial Analysis</div>
            <div class="ra-subtext">
                Analyze annual reports from leading publicly traded companies using natural language.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.space("xlarge")
    analyze_single_clicked = st.button("Analyze a Company's Report", key="start_analyzing", width="stretch")

    if analyze_single_clicked:
        st.session_state["ui_phase"] = "selecting_single_report"
        st.rerun()


    if st.session_state["ui_phase"] == "selecting_single_report":
        entries = load_report_entries()
        if not entries:
            st.error("No reports found in data/reports/.")
            return

        selected = render_report_picker_single_select(entries)
        if selected:
            st.session_state["selected_doc_id"] = selected.doc_id
            st.session_state["selected_entry"] = to_entry_dict(selected)
            st.session_state["chat_messages"] = []
            st.session_state["ui_phase"] = "chat_single_report"
            _go_to_chat_page()


main()
