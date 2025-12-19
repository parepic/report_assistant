# report_assistant

A RAG (Retrieval-Augmented Generation) system for Q&A over company documents using chunking, embeddings, and vector search.

## Setup

1. Clone the repository and navigate to the directory:
   ```
   git clone https://github.com/parepic/report_assistant.git
   cd report_assistant
   ```

2. Install Ollama (for local LLM and embeddings):
   - Download from https://ollama.ai/
   - Pull the required models:
     ```
     ollama pull phi
     ollama pull nomic-embed-text
     ```

3. Install dependencies:
   - Install PDM: https://pdm.fming.dev/latest/#installation
   - Set Python interpreter: `pdm use python` (requires Python >= 3.11)
   - Install deps: `pdm install`
   - Keep deps in sync (important when collaborators update pyproject/lock): `pdm sync`

4. Start Qdrant (vector database):
   ```
   docker-compose up -d
   ```

## Usage

Run the full pipeline (chunking, embedding, and Q&A):
```
pdm run python pipeline.py
```

This will:
- Chunk the document based on `global.yaml` config.
- Generate embeddings and store in Qdrant.
- Start an interactive Q&A session.

## Data & Output Layout

- Input data lives under `data/` with an `index.json` listing documents (doc_id, company, fiscal_year, paths).
- Chunking creates per-company output under `output/<company_slug>/`:
  - `text/<doc_id>.md`: Markdown version (preferred for preserving tables).
  - `chunks/<doc_id>.json`: chunked content plus metadata.

## Key Modules / Types

- `data_classes.py`: Pydantic models for `DocumentEntry`, `ChunkStrategy`, and `ChunkFile` (includes a hash over strategy + chunk content).
- `chunking/algorithms.py`: chunkers. `sequential` chunks by iterating over characters with configurable overlap; `sentences`, meanwhile, chunks into discrete sentences.
- `chunking/chunk.py`: orchestrates loading source files, saving plaintext, running the chunker, and writing chunk JSON.
- `chunking/convert_to_markdown.py`: converts `.docx` to Markdown. Markdown is preferable to plaintext because tables render cleanly and match LLM pretraining formats.
- `embed.py`: loads chunks, generates embeddings via Ollama, stores in Qdrant with metadata.
- `llm.py`: interactive Q&A using vector search in Qdrant and LLM generation.
- `pipeline.py`: runs chunk → embed → llm sequentially.
- `notebooks/`: exploratory notebooks (e.g., sentence chunking, data exploration).

## Converting Word Documents to Markdown

We implemented two different ways to convert word documents to a markdown file. We can compare model perfromance to see if the conversion strategy has a large impact, especially for quantitative questions on tables.

## Config
- `global.yaml` controls data/input/output paths, report_id selection, endpoints, and chunking strategy.
