from __future__ import annotations

import streamlit as st


import streamlit as st

def _render_citations(citations: list[dict]) -> None:
    """
    Renders citations as styled cards with category badges and expandable text.
    """
    if not citations:
        return

    # 1. CSS for the "Category Badge" to make it look like a UI tag
    st.markdown(
        """
        <style>
        .category-badge {
            background-color: #f0f2f6;
            color: #31333F;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid #d6d6d8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Dark mode adjustment if needed, though simpler to stick to neutral colors */
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📚 View Source Evidence", expanded=False):
        seen_factors: set[str] = set()
        
        for citation in citations:
            risk_factor = str(citation.get("risk_factor", "Detailed Disclosure"))
            
            # Deduplication: Ensure we don't show the same Risk Factor header twice
            if risk_factor in seen_factors:
                continue
            seen_factors.add(risk_factor)
            
            text = str(citation.get("text", ""))
            # Truncate text to keep the UI clean, but allow user to read enough context
            snippet = text[:600].strip() + "..." if len(text) > 600 else text

            # 2. Render as a "Card" using the native border container
            with st.container(border=True):
                # Row 1: The Category Badge
                st.markdown(f'<span class="category-badge">{risk_factor}</span>', unsafe_allow_html=True)
                
                # # Row 2: The Risk Factor Title (Bold and prominent)
                # st.markdown(f"#### {risk_factor}")
                
                # Row 3: The actual text content (Italicized for 'quote' feel)
                st.caption(f"{snippet}")
                
                # Optional: A visual indicator that this is from the 10-K
                st.markdown("**Source:** *SEC Filing 10-K*")
                
def render_chat_history(messages: list[dict], company: str | None = None, year: str | int | None = None) -> None:
    if not messages:
        if company and year:
            prompt = f"Ask any question about {company}'s {year} risk assessment report"
        elif company:
            prompt = f"Ask any question about {company}'s risk assessment report"
        else:
            prompt = "Ask any question about this risk assessment report"
        st.markdown(f"<div class='ra-chat-prompt'>{prompt}</div>", unsafe_allow_html=True)
        return

    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                _render_citations(message.get("citations", []))
