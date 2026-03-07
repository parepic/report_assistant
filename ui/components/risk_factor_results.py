from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_risk_factor_results(items: List[Dict[str, Any]]) -> None:
    """
    Render risk factors in an expandable list.

    Clicking a risk-factor header expands/collapses its full text.
    """
    count = len(items)
    st.subheader(f"Risk Factors ({count})")

    if count == 0:
        st.info("No risk factors found for this document.")
        return

    for item in items:
        idx = item.get("idx", "?")
        risk_factor = str(item.get("risk_factor") or "Untitled risk factor")
        text = str(item.get("text") or "")
        label = f"{idx}. {risk_factor}"

        with st.expander(label, expanded=False):
            st.markdown(text)
