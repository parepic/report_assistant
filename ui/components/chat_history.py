from __future__ import annotations

import streamlit as st


def render_chat_history(messages: list[dict]) -> None:
    if not messages:
        st.markdown("<div class='ra-empty'>Ask a question about this report to begin.</div>", unsafe_allow_html=True)
        if not st.session_state.get("compare_to_last_year_clicked", False):
            left, center, right = st.columns([1, 1, 1])
            with center:
                if st.button("Compare to last year", key="compare_to_last_year"):
                    st.session_state["compare_to_last_year_clicked"] = True
                    st.rerun()
        return

    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
