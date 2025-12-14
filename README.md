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

## File Explanations

- `global.yaml`: Configuration file with paths, URLs, and chunking strategy.
- `chunk.py`: Loads documents, chunks text using sequential overlapping method, saves to JSON.
- `embed.py`: Loads chunks, generates embeddings via Ollama, stores in Qdrant with metadata.
- `llm.py`: Interactive Q&A using vector search in Qdrant and LLM generation.
- `pipeline.py`: Runs chunk.py, embed.py, and llm.py sequentially.
- `notebooks/explore_data.ipynb`: Jupyter notebook for exploring Qdrant data and collections.
