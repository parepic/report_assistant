## 10-K Risk Analysis RAG Assistant

A production-style **Retrieval-Augmented Generation (RAG)** system for **10-K Risk Factors** with two core capabilities:

- **Grounded Q&A** over a selected filing (LLM answers with **citations** to source text)
- **Year-over-year (YoY) change detection** across filings (added/removed/modified risks + concise summaries)

Built with clean boundaries, repeatable ingestion, and an evaluation harness so it scales beyond a prototype.

---

## What it does

### 1) Risk Report Chatbot (grounded Q&A)
Pick a company + year and ask questions via the chat UI:
- Retrieves relevant risk-factor sections
- Generates answers via **OpenAI** with **citations** to supporting passages

### 2) YoY Risk Change Detection
Compare a filing with the previous year and detect meaningful risk disclosure changes:
- **Added paragraphs**
- **Removed paragraphs**
- **Modified paragraphs**

The UI highlights differences and generates **bullet summaries** describing how disclosed risks evolved.

---

## Tech stack

- **FastAPI** (API)
- **Qdrant** (vector database)
- **PostgreSQL** (document + metadata storage)
- **Ollama** (`nomic-embed-text`) for embeddings
- **OpenAI** for generation
- **Streamlit** UI
- **DeepEval** evaluation (LLM-as-judge metrics)

---

## Setup

1) Clone the repo:
```bash
git clone https://github.com/parepic/report_assistant.git
cd report_assistant
```

2) Configure OpenAI (primary generator):
- Set `OPENAI_API_KEY` via env var or a `.env` file in the repo root.

3) Install Ollama (embeddings):
- Install: https://ollama.ai/
- Pull the embedding model:
```bash
ollama pull nomic-embed-text
```

4) Install dependencies (Python ≥ 3.11 via PDM):
- Install PDM: https://pdm.fming.dev/latest/#installation
```bash
pdm use python
pdm install
# When collaborators update pyproject/lock:
pdm sync
```

5) Start Qdrant + PostgreSQL:
```bash
docker-compose up -d
```

---

## Usage

### Ingestion pipelines

Three pipelines cover database ingestion, chatbot indexing, and YoY indexing:

```bash
python -m app.ingestion.pipeline_db
# Parse filing → store markdown in PostgreSQL

python -m app.ingestion.pipeline_chatbot
# Chunk + embed for chatbot retrieval

python -m app.ingestion.pipeline_comparison
# Chunk + embed for YoY comparison (separate collection)
```

You can run a single stage too. For example:
```bash
pdm run python pipeline_chatbot.py --chunk
```

Or combine stages:
```bash
pdm run python pipeline_chatbot.py --chunk --embed
```

---

## UI

Run the Streamlit app:
```bash
pdm run streamlit run ui/streamlit_app.py
```

More details live in `ui/README.md`.

---

## Evaluation

Run:
```bash
pdm run deepeval_eval/eval_rag.py
```

Evaluation checks:
1) Retrieval quality (does it pull the right context?)
2) Answer quality (is the answer relevant and faithful to the context?)

We use **LLM-as-judge** metrics via DeepEval, attaching retrieved context to each test case to score both retrieval and generation. See `deepeval_eval/eval_rag.py` for an end-to-end example that mirrors the app prompt while including retrieved passages for scoring.

---

## Data & output layout

- Input data under `data/` with `index.json` (doc_id, company, fiscal_year, paths)
- Outputs per company under `output/<company_slug>/`:
  - `text/<doc_id>.md` Markdown version (keeps tables well)
  - `chunks/<doc_id>.json` chunked content + metadata

---

## Architecture

### API Layer (`api`)
- FastAPI routers for chat, comparison, and document listing
- Dependency wiring in `dep.py`

### Service Layer (`services`)
- Chat service: retrieval → prompt construction → generation
- Comparison service: YoY diffing + change summaries

### Ingestion Layer (`ingestion`)
- Parsing, markdown conversion, chunking, embedding, and indexing
- Separate entry points for DB, chatbot, and YoY flows

### Clients (`clients`)
- Thin wrappers around external services (Qdrant, OpenAI)

### Data Contracts (`data_classes.py`, `models.py`)
- Pydantic models for config and contracts
- SQLAlchemy models for persistence

### UI Boundary (`ui`)
- Streamlit UI kept separate from core application logic

---

## Config

`global.yaml` controls major components (LLM model, embedding model, chunking strategy) to speed up experimentation.

---

## Word → Markdown conversion

Two Word → Markdown conversion paths exist so you can compare downstream RAG quality, especially for table-heavy or numeric questions.
