# AGENTS.md

This file defines rules and conventions for agents
working on the UI of this repository. These rules exist to protect repository
structure, import discipline, and long-term maintainability.

If there is any ambiguity, follow these rules over assumptions.

---

## Project overview (source of truth)

`report_assistant/ui` is a simple UI built using **Streamlit** to allow non-technically adept users to analyze reports in a browser. The UI experience should resemble that of using ChatGPT. The UI is an internal / research interface, not a consumer product.

Main features:
- Selecting company 10K Reports from our `data/reports` database
- Adding company reports to the database
- Selecting a report and entering chat mode, where the user ask questions about the company


The Streamlit entrypoint is:
```bash
streamlit run ui/streamlit_app.py
```

---

## 1. UI Location and Scope

- All UI code must live inside the `ui/` directory
- No UI files are allowed in the repository root
- The UI is a thin presentation layer only
- Every page must have a corresponding file in the `ui/pages/` directory

Expected structure (approximate):
```
ui/
├── streamlit_app.py # Entry point
├── pages/ # Streamlit pages
├── components/ # Reusable UI components
├── assets/ # CSS, images, static files
└── README.md
```


## 2. Import Discipline
### Allowed imports
- UI code MAY import from the core project package (e.g. `your_project.*`)
- Do not install any packages yourself, rather ask me to install them myself before continuing.
- UI code MUST NOT use `sys.path` manipulation


## 3. Session State Usage

Streamlit st.session_state is the single source of truth for:
- selected report / document ID
- chat history
- UI navigation state

Session state keys must:
- be explicitly initialized
- use descriptive, stable names

Example (illustrative only):
```python
st.session_state["selected_doc_id"]
st.session_state["chat_messages"]
```

## 4. Styling Guidelines
Styling should be minimal and neutral

Prefer a ChatGPT-like visual style:
- rounded elements
- neutral background
- clean sans-serif typography
- Custom CSS must live in ui/assets/


## 5. Streamlit HTML composition
- Raw HTML passed via `st.markdown(..., unsafe_allow_html=True)` must be self-contained
- Do NOT open an HTML tag in one Streamlit call and close it in another
- Streamlit components (`st.button`, `st.columns`, `st.container`, etc.) are may be freely composed

✅ Allowed:
st.columns([3, 1])
st.button("Analyze")

❌ Not allowed:
st.markdown("<div>", unsafe_allow_html=True)
st.button("Analyze")
st.markdown("</div>", unsafe_allow_html=True)