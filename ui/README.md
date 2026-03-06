# Report Assistant UI (MVP)

Run the Streamlit app from the repo root:
```
pdm run streamlit run ui/streamlit_app.py
```

Prereqs:
- `OPENAI_API_KEY` is set
- Qdrant is running and embeddings already exist for the selected report
- Ollama is running for query embeddings

Notes:
- Reports are loaded only from `app/data/reports/` and indexed via `app/data/index.json`
