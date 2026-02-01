from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_file_browser import st_file_browser

from services.documents import get_config, report_display_name, report_metadata, to_entry_dict
from report_assistant.data_classes import DocumentEntry




def render_report_picker_single_select(entries: list[DocumentEntry]) -> Optional[DocumentEntry]:

    @st.dialog("Select a Report to Analyze", width="medium")
    def _modal() -> Optional[DocumentEntry]:

        st.markdown("<div class='ra-modal-header'>Select a report to analyze</div>", unsafe_allow_html=True)
        st.markdown("<div class='ra-modal-subtitle'>Choose one document to continue</div>", unsafe_allow_html=True)


        selected: Optional[DocumentEntry] = None
        for entry in entries:
            display_name = report_display_name(entry)
            metadata = report_metadata(entry)
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                st.markdown(
                    f"""
                    <div class="ra-report-hoverable">
                        <strong>{display_name}</strong><br/>
                        <span class="ra-report-metadata">{metadata}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Analyze", key=f"select_{entry.doc_id}", width="stretch"):
                    selected = entry

                    st.session_state["selected_reports"] = [to_entry_dict(selected)]
                    st.session_state["chat_messages"] = []
                    st.session_state["ui_phase"] = "chat_single_report"
                    
                    st.switch_page("pages/chat.py")






            st.divider()
        return selected
    
    return _modal()






def render_report_picker_multi_select(
    entries: list[DocumentEntry],
) -> Optional[list[DocumentEntry]]:

    @st.dialog("Select a Report to Analyze", width="medium")
    def _modal() -> Optional[list[DocumentEntry]]:

        st.markdown("<div class='ra-modal-header'>Select multiple reports to compare</div>", unsafe_allow_html=True)
        st.markdown("<div class='ra-modal-subtitle'>Choose two or more reports to continue</div>", unsafe_allow_html=True)

        reports_root = Path(get_config().data_path) / "reports"
        entries_by_path = {
            Path(entry.source_file_path).resolve(): entry for entry in entries
        }

        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            compare_clicked = st.button("Compare", width="stretch")



        browser_event = st_file_browser(
            path=str(reports_root),
            key="report_file_browser",
            show_preview=False,
            show_download_file=False,
        )
        if (
            not compare_clicked
            and isinstance(browser_event, dict)
            and browser_event.get("type") == "SELECT_FILE"
            and browser_event.get("target", {}).get("path")
        ):
            selected_relative_path = browser_event["target"]["path"]

            selected_reports = st.session_state.get("selected_reports", [])
            selected_reports_by_id = {
                report["doc_id"]: report for report in selected_reports
            }
            clicked_entry = entries_by_path.get(
                (reports_root / selected_relative_path).resolve()
            )
            if clicked_entry:
                if clicked_entry.doc_id in selected_reports_by_id:
                    selected_reports_by_id.pop(clicked_entry.doc_id, None)
                else:
                    selected_reports_by_id[clicked_entry.doc_id] = to_entry_dict(
                        clicked_entry
                    )
            st.session_state["selected_reports"] = list(selected_reports_by_id.values())

        selected_entries = [
            DocumentEntry.model_validate(report)
            for report in st.session_state.get("selected_reports", [])
        ]

        with col1:
            st.caption(f"Selected reports: {len(selected_entries)}")
            if selected_entries:
                st.markdown(
                    "\n".join(f"- {report_display_name(entry)}" for entry in selected_entries)
                )



        if compare_clicked:
            if len(selected_entries) < 2:
                st.warning("Select at least two reports to compare.")
                return None
            
            st.session_state["selected_reports"] = [
                to_entry_dict(entry) for entry in selected_entries
            ]
            st.session_state["chat_messages"] = []
            st.session_state["ui_phase"] = "chat_multi_report"
            
            st.switch_page("pages/chat.py")



        return selected_entries
    
    return _modal()


