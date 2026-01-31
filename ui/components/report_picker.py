from __future__ import annotations

from typing import Optional

import streamlit as st

from services.documents import report_display_name, report_metadata
from report_assistant.data_classes import DocumentEntry


def render_report_picker(entries: list[DocumentEntry]) -> Optional[DocumentEntry]:
    st.markdown(
        """
        <div class="ra-modal-overlay"></div>
        <div class="ra-modal-card">
            <div class="ra-modal-header">Select a report to analyze</div>
            <div class="ra-modal-subtitle">Choose one document to continue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected: Optional[DocumentEntry] = None
    container = st.container()
    with container:
        for entry in entries:
            cols = st.columns([6, 3, 2])
            with cols[0]:
                st.markdown(f"<div class='ra-report-name'>{report_display_name(entry)}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div class='ra-report-meta'>{report_metadata(entry)}</div>", unsafe_allow_html=True)
            with cols[2]:
                if st.button("Analyze", key=f"analyze_{entry.doc_id}"):
                    selected = entry
            st.markdown("<div class='ra-divider'></div>", unsafe_allow_html=True)

    return selected
