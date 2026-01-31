# AGENTS.md

This file defines non-negotiable rules and conventions for agents
working on the UI of this repository. These rules exist to protect repository
structure, import discipline, and long-term maintainability.

If there is any ambiguity, follow these rules over assumptions.

---

## Project overview (source of truth)

`report_assistant/ui` is a simple UI developed to allow non-technically adept users to analyze reports in a browser. The UI experience should resemble that of using ChatGPT.

Main features:
- Selecting company 10K Reports from our `data/reports` database
- Adding company reports to the database
- Selecting a report and entering chat mode, where the user ask questions about the company


The Streamlit entrypoint is:
```bash
streamlit run ui/streamlit_app.py
```

---

## 1. UI Framework Choice
- The UI must be built using **Streamlit**
- Do NOT introduce backend servers (FastAPI, Flask, etc.) for the UI
- The UI is an internal / research interface, not a consumer product

Frameworks for styling: 

---

## 2. UI Location and Scope

- All UI code must live inside the `ui/` directory
- No UI files are allowed in the repository root
- The UI is a thin presentation layer only

Expected structure (approximate):
```
ui/
├── streamlit_app.py # Entry point
├── pages/ # Streamlit pages
├── components/ # Reusable UI components
├── assets/ # CSS, images, static files
└── README.md
```


## 3. Import Discipline
### Allowed imports
- UI code MAY import from the core project package (e.g. `your_project.*`)
- Do not install any packages yourself, rather ask me to install them myself before continuing.

### Forbidden imports
- Core project code MUST NOT import from `ui/`
- UI code MUST NOT modify or monkey-patch core logic
- UI code MUST NOT use `sys.path` manipulation


## 4. Session State Usage

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

## 5. Styling Guidelines
Styling should be minimal and neutral

Prefer a ChatGPT-like visual style:
- rounded elements
- neutral background
- clean sans-serif typography
- Custom CSS must live in ui/assets/
- Inline HTML/CSS should be minimal and justified