from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from services.documents import load_report_entries


def render_report_upload_modal() -> None:
    @st.dialog("Add Your Own Report", width="medium")
    def _upload_modal() -> None:
        st.markdown(
            "<div class='ra-modal-header'>Add your own report</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ra-modal-subtitle'>Provide report details to add it to the library.</div>",
            unsafe_allow_html=True,
        )

        entries = load_report_entries()
        company_options = sorted({entry.company for entry in entries})

        with st.form("upload_report_form"):
            file_id = st.text_input("File ID name")
            year = st.number_input("Year published", min_value=1900, max_value=2100, step=1)
            company = st.selectbox(
                "Company",
                options=company_options,
                index=None,
                placeholder="Start typing to search",
            )
            company_other = st.text_input("Company (if not listed)")
            upload = st.file_uploader("Upload a .docx report", type=["docx"])
            submitted = st.form_submit_button("Submit")

        if submitted:
            chosen_company = company_other.strip() or (company or "").strip()
            if not file_id.strip():
                st.error("File ID name is required.")
                return
            if not chosen_company:
                st.error("Company name is required.")
                return
            if upload is None:
                st.error("Please upload a .docx report.")
                return

            reports_root = Path("data/reports")
            target_dir = reports_root / chosen_company / file_id.strip()
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{file_id.strip()}.docx"
            with target_path.open("wb") as handle:
                handle.write(upload.getbuffer())

            index_path = Path("data/index.json")
            if index_path.is_file():
                index_data = json.loads(index_path.read_text(encoding="utf-8"))
            else:
                index_data = []

            if any(entry.get("doc_id") == file_id.strip() for entry in index_data):
                st.error("A report with this File ID already exists.")
                return

            index_data.append(
                {
                    "doc_id": file_id.strip(),
                    "company": chosen_company,
                    "fiscal_year": int(year),
                    "source_file_path": str(target_path.as_posix()),
                    "questions_file_path": None,
                }
            )
            index_path.write_text(
                json.dumps(index_data, indent=2),
                encoding="utf-8",
            )
            st.success("Report added to index.")

    _upload_modal()
