<!--
AGENTS.md (repo root)

This file defines conventions for AI coding agents working in this repository.
If something is unclear, ask before making assumptions or large refactors.
-->

# Agent Guide: report_assistant

## Project overview (source of truth)

`report_assistant` is a small RAG pipeline for Q&A over company documents:

1. **Chunking**: Convert `.docx` → Markdown, then chunk based on `global.yaml`.
2. **Embedding**: Embed chunks (via **Ollama embeddings**) and store in **Qdrant**.
3. **Answering**: Retrieve top-k chunks from Qdrant and generate answers via **OpenAI API**.
4. **Evaluation / sanity checks**: Run question sets and/or DeepEval scripts.

The main orchestration is `pipeline.py`.

## Non-negotiable rules

### 1) LLM provider policy (OpenAI-first)
- Treat **OpenAI API** as the primary provider for **generation** (`report_assistant/llm.py`).
- Ollama may remain as a **fallback artifact** in the codebase, but do not migrate core generation back to Ollama unless explicitly requested.
- Embeddings are currently produced via **Ollama** (see `report_assistant/embed.py` and `report_assistant/llm.py:get_embedding`).

### 2) Dependency / install policy (ask first)
- Do **not** add, remove, or upgrade dependencies without asking first.
- Do **not** edit `pyproject.toml` or `pdm.lock` unless the user explicitly approves.
- If you need a new package, stop and ask the user to install/approve it before continuing.

### 3) UI boundary
- All UI code must live under `ui/`.
- Core code under `report_assistant/` must **not** import from `ui/`.
- Follow the stricter UI rules in `ui/AGENTS.override.md` when touching any `ui/` files.

### 4) Don’t break the data contract
- `data/index.json` and `global.yaml` are treated as the configuration interface.
- Keep `DocumentEntry` / `GlobalConfig` fields in `report_assistant/data_classes.py` consistent with how `load_utils.py` loads them.
- Qdrant collection names should be created via `report_assistant.utils.utils.slugify_name()` (format: `company__<slug>`).
- Retrieval can be filtered by `strategy_hash` (see `compute_strategy_hash`).

## How to run (agent-friendly)

### Prereqs
- Python via PDM (see `pyproject.toml`)
- Qdrant via `docker-compose.yml`
- **OpenAI API key** available as `OPENAI_API_KEY` (env var or `.env`)
- Ollama running (for embeddings) and embedding model pulled (default: `nomic-embed-text`)

### Typical commands
- Full pipeline: `pdm run python pipeline.py`
- Stages: `pdm run python pipeline.py --chunk --embed --test` (flags can be combined)

## Code conventions (keep it boring)
- Prefer minimal, surgical changes over refactors.
- Keep functions small and typed; match existing style (Pydantic v2 models, simple modules).
- Avoid `sys.path` hacks (exception: evaluation scripts may need repo-root import handling; keep it localized).
- Prefer updating docs/config examples when behavior changes.

## What “tests” mean here
- `tests/test.py` is currently an **evaluation / sanity-check runner**, not strict `pytest` unit tests.
- Don’t invest in heavy test infra unless requested; keep evaluation flows working.
