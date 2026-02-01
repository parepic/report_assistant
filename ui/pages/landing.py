from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.components.report_picker_amar import render_report_picker_multi_select, render_report_picker_single_select
from ui.components.report_upload_modal import render_report_upload_modal
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
    analyze_single_clicked = st.button("Analyze a Company's Report", width="stretch")

    if analyze_single_clicked:
        st.session_state["ui_phase"] = "selecting_single_report"
        st.rerun()


    if st.session_state["ui_phase"] == "selecting_single_report":
        entries = load_report_entries()
        if not entries:
            st.error("No reports found in data/reports/.")
            return

        render_report_picker_single_select(entries)


    


    st.space("small")
    analyze_multi_clicked = st.button("Compare Multiple Reports", width="stretch")

    if analyze_multi_clicked:
        st.session_state["ui_phase"] = "selecting_multi_report"
        st.rerun()

    if st.session_state["ui_phase"] == "selecting_multi_report":
        entries = load_report_entries()
        if not entries:
            st.error("No reports found in data/reports/.")
            return

        render_report_picker_multi_select(entries)




    st.space("small")
    upload_clicked = st.button("Add Your Own Report", width="stretch")

    if upload_clicked:
        st.session_state["ui_phase"] = "upload"
        st.rerun()

    if st.session_state["ui_phase"] == "upload":
        render_report_upload_modal()


main()
