# Report Assistant UI (MVP)

Run the Streamlit app from the repo root:
```
PYTHONPATH=. streamlit run ui/streamlit_app.py
```

Prereqs:
- `OPENAI_API_KEY` is set
- Qdrant is running and embeddings already exist for the selected report
- Ollama is running for query embeddings

Notes:
- Reports are loaded only from `data/reports/` and indexed via `data/index.json`
- This UI is a thin layer over `report_assistant/` and does not modify core logic
