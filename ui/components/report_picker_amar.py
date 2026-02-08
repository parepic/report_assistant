from __future__ import annotations

from typing import Optional

import streamlit as st

from services.documents import report_display_name, report_metadata, to_entry_dict




def render_report_picker_single_select(entries: list[dict]) -> Optional[dict]:

    @st.dialog("Select a Report to Analyze", width="medium")
    def _modal() -> Optional[dict]:

        st.markdown("<div class='ra-modal-header'>Select a report to analyze</div>", unsafe_allow_html=True)
        st.markdown("<div class='ra-modal-subtitle'>Choose one document to continue</div>", unsafe_allow_html=True)


        selected: Optional[dict] = None
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
                if st.button("Analyze", key=f"select_{entry.get('doc_id')}", width="stretch"):
                    selected = entry

                    st.session_state["selected_reports"] = [to_entry_dict(selected)]
                    st.session_state["chat_messages"] = []
                    st.session_state["ui_phase"] = "chat_single_report"
                    
                    st.switch_page("pages/chat.py")






            st.divider()
        return selected
    
    return _modal()






def render_report_picker_multi_select(
    entries: list[dict],
) -> Optional[list[dict]]:

    @st.dialog("Select a Report to Analyze", width="medium")
    def _modal() -> Optional[list[dict]]:

        st.markdown("<div class='ra-modal-header'>Select multiple reports to compare</div>", unsafe_allow_html=True)
        st.markdown("<div class='ra-modal-subtitle'>Choose two or more reports to continue</div>", unsafe_allow_html=True)

        entries_by_id = {str(entry.get("doc_id")): entry for entry in entries}
        options = list(entries_by_id.keys())

        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            compare_clicked = st.button("Compare", width="stretch")

        selected_ids = st.multiselect(
            "Select reports",
            options=options,
            default=[str(report.get("doc_id")) for report in st.session_state.get("selected_reports", [])],
            format_func=lambda doc_id: report_display_name(entries_by_id.get(str(doc_id), {})),
        )

        selected_entries = [entries_by_id[str(doc_id)] for doc_id in selected_ids if str(doc_id) in entries_by_id]
        st.session_state["selected_reports"] = [to_entry_dict(entry) for entry in selected_entries]

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

            st.session_state["chat_messages"] = []
            st.session_state["ui_phase"] = "chat_multi_report"
            st.switch_page("pages/chat.py")

        return selected_entries
    
    return _modal()


