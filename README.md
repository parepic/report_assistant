# report_assistant

A RAG (Retrieval-Augmented Generation) system for Q&A over company documents using chunking, embeddings, and vector search.

## Setup

1. Clone the repository and navigate to the directory:
   ```
   git clone https://github.com/parepic/report_assistant.git
   cd report_assistant
   ```

2. Set up OpenAI access (primary LLM for generation):
   - Set `OPENAI_API_KEY` (env var or a `.env` file in the repo root).
   - The model is configured via `LLM_MODEL` in `global.yaml` (default: `gpt-4o-mini`).

3. Install Ollama (used for **embeddings**; generation is OpenAI-first):
   - Download from https://ollama.ai/
   - Pull the required embedding model:
     ```
     ollama pull nomic-embed-text
     ```
   - Optional: you can also pull a local chat model (e.g. `llama3.1:8b`) if you want to experiment with local judging or fallback scripts.

4. Install dependencies:
   - Install PDM: https://pdm.fming.dev/latest/#installation
   - Set Python interpreter: `pdm use python` (requires Python >= 3.11)
   - Install deps: `pdm install`
   - Keep deps in sync (important when collaborators update pyproject/lock): `pdm sync`

5. Start Qdrant (vector database):
   ```
   docker-compose up -d
   ```


## Usage

Run the full pipeline (chunking, embedding, and testing):
```
pdm run python pipeline.py
```

This will:
- Chunk the document based on `global.yaml` config.
- Generate embeddings (via Ollama) and store in Qdrant.
- Run an evaluation / sanity-check pass to compare LLM outputs with expected answers (when a questions file exists).

To run individual stages, use these flags:
```
pdm run python pipeline.py --chunk   # Run chunking only
pdm run python pipeline.py --embed   # Run embedding only
pdm run python pipeline.py --test    # Run testing only
pdm run python pipeline.py --llm     # Run interactive Q&A session
```

You can combine multiple flags to run specific stages in sequence:
```
pdm run python pipeline.py --chunk --embed
```


## UI
To run the app in a GUI with streamlit, run:
```bash
pdm run streamlit run ui/streamlit_app.py
```

For more informaysh check the [UI-specific README.md](ui/README.md)

## Evaluation

To run the evaluation, type the following into the terminal:
```bash
pdm run deepeval_eval/eval_rag.py
```

Evaluating a RAG system means testing two things: (1) whether retrieval finds the right context, and (2) whether the model answers correctly using that context. We use an **LLM-as-judge** approach, where a separate model scores outputs with metrics that quantify retrieval quality (context relevance) and answer quality (relevance/faithfulness to context).

In this repo, DeepEval is used to run those metrics. The high-level flow is:
1) Load a question set with expected answers.
2) Run the real RAG pipeline (retrieve → build prompt → generate answer).
3) Build DeepEval test cases that include the retrieved context.
4) Run metrics for retrieval (e.g., contextual precision/recall) and answering (e.g., answer relevancy/faithfulness).

See `deepeval_eval/eval_rag.py` for a minimal, end-to-end example that mirrors the app’s RAG prompt while attaching retrieval context for evaluation.

Run the file to see how the LLM-as-judge performs. In practice, we’ve moved the main app generation to OpenAI, and we generally recommend using a strong judge model (OpenAI or a flagship Google model) rather than a small local model for evaluation. Ollama can still be used for local experiments, but treat it as a fallback.


## Data & Output Layout

- Input data lives under `data/` with an `index.json` listing documents (doc_id, company, fiscal_year, paths).
- Chunking creates per-company output under `output/<company_slug>/`:
  - `text/<doc_id>.md`: Markdown version (preferred for preserving tables).
  - `chunks/<doc_id>.json`: chunked content plus metadata.

## Key Modules / Types

- `data_classes.py`: Pydantic models for `DocumentEntry`, `ChunkStrategy`, and `ChunkFile` (includes a hash over strategy + chunk content).
- `chunking/chunk.py`: orchestrates loading source files, saving plaintext, running the chunker, and writing chunk JSON.
- `chunking/convert_to_markdown.py`: converts `.docx` to Markdown. Markdown is preferable to plaintext because tables render cleanly and match LLM pretraining formats.
- `embed.py`: loads chunks, generates embeddings via Ollama, stores in Qdrant with metadata.
- `llm.py`: interactive Q&A using vector search in Qdrant and LLM generation.
- `test.py`: Contains testing logic, where llm answers are compared with expected answers.
- `pipeline.py`: runs chunk → embed → llm sequentially.
- `notebooks/`: exploratory notebooks (e.g., sentence chunking, data exploration).
- `chunking/strategies`: Chunking strategy classes live in this dir. Each strategy class inherits from `ChunkStrategy` base class.

## Converting Word Documents to Markdown

We implemented two different ways to convert word documents to a markdown file. We can compare model perfromance to see if the conversion strategy has a large impact, especially for quantitative questions on tables.

## Config
- `global.yaml` controls data/input/output paths, report_id selection, endpoints, and chunking strategy.
