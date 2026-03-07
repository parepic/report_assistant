from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.chat_history import render_chat_history
from components.chat_input import get_user_input
from components.compare_results import render_compare_results
from components.risk_factor_results import render_risk_factor_results
from services.documents import from_entry_dict, report_display_name
from services.rag import answer_for_entry
from services.comparison import compare_to_last_year
from services.risk_factors import browse_risk_factors
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


@st.dialog("Comparison Summary")
def _show_compare_dialog(result: dict) -> None:
    with st.container(height=800):
        render_compare_results(result)


@st.dialog("Risk Factors")
def _show_risk_factors_dialog(items: list[dict]) -> None:
    with st.container(height=800):
        render_risk_factor_results(items)


def main() -> None:
    init_session_state()
    _inject_css()

    selected_reports = st.session_state.get("selected_reports") or []

    if len(selected_reports) >= 2:
        entries = [
            from_entry_dict(entry) if isinstance(entry, dict) else entry
            for entry in selected_reports
        ]
        st.markdown(
            """
            <div class="ra-context">
                <div class="ra-context-title">Comparing multiple reports</div>
                <div class="ra-context-subtitle">Multi-report chat is not implemented yet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for entry in entries:
            st.markdown(f"- {report_display_name(entry)}")
        st.info("Select a single report to start chatting.")
        return

    if not selected_reports:
        st.error("Select a report before entering chat.")
        if st.button("Go to Start"):
            _go_to_start()
        return

    entry = selected_reports[0]
    if isinstance(entry, dict):
        entry = from_entry_dict(entry)
    company = str(entry.get("company", "")).strip() or "Unknown"
    year = entry.get("fiscal_year", "?")
    with st.sidebar:
        st.subheader("Active Document")
        st.markdown(
            f"""
            <div class="ra-active-doc">
                <div class="ra-doc-company">Company: {company}</div>
                <div class="ra-doc-year" style="font-weight: 700;">FY {year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Document Actions")
        if st.button("✨ Compare to Prev. Fiscal Year", key="compare_to_prev_fiscal_year"):
            st.session_state["compare_to_last_year_clicked"] = True
            st.rerun()
        if st.button("✨ Browse Risk Factor", key="browse_risk_factor"):
            st.session_state["browse_risk_factors_result"] = None
            st.session_state["browse_risk_factors_clicked"] = True
            st.rerun()

    messages = st.session_state["chat_messages"]
    render_chat_history(messages, company=company, year=year)

    if st.session_state.get("compare_to_last_year_clicked"):
        if st.session_state.get("compare_to_last_year_result") is None:
            with st.spinner("Comparing to last year..."):
                try:
                    st.session_state["compare_to_last_year_result"] = compare_to_last_year(
                        doc_id=str(entry.get("doc_id"))
                    )
                except Exception as exc:
                    st.session_state["compare_to_last_year_result"] = {
                        "changed": [],
                        "added": [],
                        "removed": [],
                    }
                    st.error(f"Compare to last year failed: {exc}")
        result = st.session_state.get("compare_to_last_year_result")
        st.session_state["compare_to_last_year_clicked"] = False
        if isinstance(result, dict):
            _show_compare_dialog(result)

    if st.session_state.get("browse_risk_factors_clicked"):
        if st.session_state.get("browse_risk_factors_result") is None:
            with st.spinner("Loading risk factors..."):
                try:
                    st.session_state["browse_risk_factors_result"] = browse_risk_factors(
                        doc_id=str(entry.get("doc_id"))
                    )
                except Exception as exc:
                    st.session_state["browse_risk_factors_result"] = []
                    st.error(f"Browse risk factors failed: {exc}")
        items = st.session_state.get("browse_risk_factors_result")
        st.session_state["browse_risk_factors_clicked"] = False
        if isinstance(items, list):
            _show_risk_factors_dialog(items)

    user_input = get_user_input()
    if user_input:
        messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                response = answer_for_entry(user_input, entry)
            except Exception as exc:
                response = {"error": f"Something went wrong while retrieving an answer: {exc}"}
        if isinstance(response, dict) and response.get("error"):
            messages.append({"role": "assistant", "content": response["error"]})
        elif isinstance(response, dict):
            messages.append(
                {
                    "role": "assistant",
                    "content": str(response.get("llm_response", "")),
                    "citations": response.get("citations", []),
                }
            )
        else:
            messages.append({"role": "assistant", "content": str(response)})
        st.rerun()


main()
