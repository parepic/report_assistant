from __future__ import annotations

from typing import Any
import html

import streamlit as st


def _inject_compare_css() -> None:
    st.markdown(
        """
        <style>
        .ra-compare-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 12px 0 8px 0;
        }
        .ra-compare-title {
            font-weight: 600;
            font-size: 1.05rem;
        }
        .ra-chip {
            display: inline-flex;
            align-items: center;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            background: #f2f2f2;
            color: #333;
            margin-left: 6px;
        }
        .ra-chip.changed { background: #eef2ff; color: #3730a3; }
        .ra-chip.added { background: #ecfeff; color: #0e7490; }
        .ra-chip.removed { background: #fef2f2; color: #b91c1c; }
        .ra-card {
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 10px 12px;
            margin: 8px 0;
            background: #fff;
        }
        .ra-accordion {
            border: 0;
            margin: 0;
        }
        .ra-accordion summary {
            list-style: none;
            cursor: pointer;
        }
        .ra-accordion summary::-webkit-details-marker {
            display: none;
        }
        .ra-card-header {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: space-between;
            font-size: 0.95rem;
        }
        .ra-card-actions {
            margin-top: 6px;
        }
        .ra-card-title {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }
        .ra-badge {
            padding: 2px 8px;
            border-radius: 8px;
            background: #f3f4f6;
            font-size: 0.8rem;
            color: #111827;
            white-space: nowrap;
        }
        .ra-block-label {
            font-weight: 600;
            margin: 8px 0 4px 0;
        }
        .ra-compare-text {
            background: #fafafa;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #eee;
            white-space: pre-wrap;
        }
        .ra-text-toggle {
            display: block;
            margin: 6px 0 0 0;
            color: #2563eb;
            cursor: pointer;
            font-size: 0.85rem;
            user-select: none;
        }
        .ra-text-toggle .less { display: none; }
        .ra-text-block input[type="checkbox"] { display: none; }
        .ra-text-block .ra-compare-text {
            max-height: 180px;
            overflow: hidden;
        }
        .ra-text-block input[type="checkbox"]:checked ~ .ra-compare-text {
            max-height: none;
        }
        .ra-text-block input[type="checkbox"]:checked ~ .ra-text-toggle .more {
            display: none;
        }
        .ra-text-block input[type="checkbox"]:checked ~ .ra-text-toggle .less {
            display: inline;
        }
        .ra-controls {
            display: flex;
            gap: 16px;
            align-items: center;
            margin: 10px 0 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _build_text_block_html(
    label: str,
    text: str,
    risk_factor: str | None,
    key_prefix: str,
) -> str:
    label_text = label
    if risk_factor:
        label_text = f"{label} — Risk factor: {html.escape(risk_factor)}"
    escaped_text = html.escape(text)
    checkbox_id = f"{key_prefix}_toggle"
    return (
        "<div class='ra-text-block'>"
        f"<div class='ra-block-label'>{label_text}</div>"
        f"<input type='checkbox' id='{checkbox_id}' />"
        f"<div class='ra-compare-text'>{escaped_text}</div>"
        f"<label class='ra-text-toggle' for='{checkbox_id}'>"
        "<span class='more'>Show more</span>"
        "<span class='less'>Show less</span>"
        "</label>"
        "</div>"
    )


def _flatten_items(change_result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in change_result.get("changed", []):
        new_data = item.get("new", {})
        old_data = item.get("removed") or item.get("old", {})
        items.append(
            {
                "type": "Changed",
                "similarity": item.get("similarity"),
                "risk_category": new_data.get("risk_category") or "N/A",
                "risk_factor": new_data.get("risk_factor") or old_data.get("risk_factor") or "N/A",
                "new_text": new_data.get("text", ""),
                "old_text": old_data.get("text", ""),
                "old_risk_factor": old_data.get("risk_factor") or "N/A",
                "llm_output": item.get("llm_output", ""),
            }
        )
    for item in change_result.get("added", []):
        new_data = item.get("added") or item.get("new", {})
        items.append(
            {
                "type": "Added",
                "similarity": item.get("similarity"),
                "risk_category": new_data.get("risk_category") or "N/A",
                "risk_factor": new_data.get("risk_factor") or "N/A",
                "new_text": new_data.get("text", ""),
                "old_text": "",
                "old_risk_factor": "N/A",
                "llm_output": item.get("llm_output", ""),
            }
        )
    for item in change_result.get("removed", []):
        old_data = item.get("removed") or item.get("old", {})
        items.append(
            {
                "type": "Removed",
                "similarity": item.get("similarity"),
                "risk_category": old_data.get("risk_category") or "N/A",
                "risk_factor": old_data.get("risk_factor") or "N/A",
                "new_text": "",
                "old_text": old_data.get("text", ""),
                "old_risk_factor": old_data.get("risk_factor") or "N/A",
                "llm_output": item.get("llm_output", ""),
            }
        )
    return items


def _sort_items(items: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    def similarity_value(item: dict[str, Any]) -> float:
        value = item.get("similarity")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    if mode == "Changed first (then Added/Removed)":
        type_order = {"Changed": 0, "Added": 1, "Removed": 2}
        return sorted(
            items,
            key=lambda item: (type_order.get(item["type"], 2), -similarity_value(item)),
        )
    if mode == "Similarity: High → Low":
        return sorted(items, key=lambda item: -similarity_value(item))
    if mode == "Similarity: Low → High":
        return sorted(items, key=lambda item: similarity_value(item))
    return items


def _format_similarity(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "Similarity: 0%"
    percent = int(round(score * 100))
    return f"Similarity: {percent}%"


def render_compare_results(change_result: dict[str, Any]) -> None:
    _inject_compare_css()

    changed_count = len(change_result.get("changed", []))
    added_count = len(change_result.get("added", []))
    removed_count = len(change_result.get("removed", []))

    st.markdown(
        f"""
        <div class="ra-compare-header">
            <div class="ra-compare-title">Document A (Year) vs Document B (Year)</div>
            <div>
                <span class="ra-chip changed">Changed ({changed_count})</span>
                <span class="ra-chip added">Added ({added_count})</span>
                <span class="ra-chip removed">Removed ({removed_count})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_cols = st.columns([2, 2])
    with control_cols[0]:
        filter_mode = st.radio(
            "Filter",
            ["All", "Changed", "Added", "Removed"],
            horizontal=True,
            index=0,
            key="compare_filter_mode",
        )
    with control_cols[1]:
        sort_mode = st.selectbox(
            "Sort",
            [
                "Changed first (then Added/Removed)",
                "Similarity: High → Low",
                "Similarity: Low → High",
            ],
            index=0,
            key="compare_sort_mode",
        )

    items = _flatten_items(change_result)
    if filter_mode in {"Changed", "Added", "Removed"}:
        items = [item for item in items if item["type"] == filter_mode]

    items = _sort_items(items, sort_mode)

    page_size = 50
    if len(items) > 200:
        page = st.slider(
            "Items",
            min_value=1,
            max_value=((len(items) - 1) // page_size) + 1,
            value=1,
            step=1,
            key="compare_page",
        )
        start = (page - 1) * page_size
        end = start + page_size
        items = items[start:end]

    if not items:
        st.info("No results to display.")
        return

    for idx, item in enumerate(items):
        type_class_map = {
            "Changed": "changed",
            "Added": "added",
            "Removed": "removed",
        }
        type_class = type_class_map.get(item["type"], "changed")
        similarity_text = _format_similarity(item.get("similarity"))
        risk_category = _truncate(item.get("risk_category", "N/A"), 80)
        risk_category_html = html.escape(risk_category)
        new_block = ""
        old_block = ""
        if item["type"] in {"Changed", "Added"}:
            label = "New" if item["type"] == "Changed" else "Added"
            new_block = _build_text_block_html(
                label,
                item.get("new_text", ""),
                item.get("risk_factor"),
                f"compare_{idx}_new",
            )
        if item["type"] in {"Changed", "Removed"}:
            label = "Previous" if item["type"] == "Changed" else "Removed"
            old_block = _build_text_block_html(
                label,
                item.get("old_text", ""),
                item.get("old_risk_factor"),
                f"compare_{idx}_old",
            )

        llm_output = item.get("llm_output")
        llm_block = ""
        if llm_output:
            llm_block = _build_text_block_html(
                "LLM insights",
                llm_output,
                None,
                f"compare_{idx}_llm",
            )

        details_html = (
            "<details class='ra-accordion'>"
            "<summary>"
            "<div class='ra-card'>"
            "<div class='ra-card-header'>"
            f"<span class='ra-chip {type_class}'>{item['type']}</span>"
            f"<div class='ra-card-title'>{risk_category_html}</div>"
            f"<span class='ra-badge'>{similarity_text}</span>"
            "</div>"
            "</div>"
            "</summary>"
            f"{new_block}"
            f"{old_block}"
            f"{llm_block}"
            "<hr />"
            "</details>"
        )
        st.markdown(details_html, unsafe_allow_html=True)
