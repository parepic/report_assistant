from __future__ import annotations

from typing import Optional

import streamlit as st

from services.documents import report_display_name, report_metadata
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
                    st.session_state["ui_phase"] = "chat_single_report"
                    st.session_state["selected_report"] = entry
                    st.experimental_rerun()
            st.divider()
        return selected
    
    return _modal()



def render_report_picker_multi_select(entries: list[DocumentEntry]) -> Optional[DocumentEntry]:

    @st.dialog("Select a Report to Analyze", width="medium")
    def _modal() -> Optional[DocumentEntry]:

        st.markdown("<div class='ra-modal-header'>Select multiple reports to compare</div>", unsafe_allow_html=True)


        selected: list[DocumentEntry] = None
        
        return selected
    
    return _modal()