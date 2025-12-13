# report_assistant
LLM tool for investor-style Q&A over annual/quarterly filings and earnings call transcripts, with chunking/embeddings for RAG, cited sources, and basic evaluation.

## Data layout
- `data/reports/<Company>/<file>`: canonical filings (annual, 10-K/10-Q).
- `data/calls/<Company>/<file>`: earnings call transcripts.
- `data/questions/<Company>/<file>`: question sets tied to a specific `doc_id` (use the manifest for the mapping).
- `data/index/documents.json`: manifest mapping each `doc_id` to its source file, type, format, questions/eval files, and all processing outputs (text, chunks, embeddings, vector store).
- `data/processed/<company>/<doc_id>/text/`: raw text extraction outputs (e.g., `pages.jsonl` with `page_num`, `text`).
- `data/processed/<company>/<doc_id>/chunks/`: chunked text ready for embedding (e.g., `chunks.jsonl` with `chunk_id`, `page_span`, `section`, `text`, `token_count`).
- `data/processed/<company>/<doc_id>/embeddings/`: serialized embeddings with text + metadata kept alongside vectors (e.g., Parquet/JSONL with `chunk_id`, `text`, `embedding`, metadata).
- `data/processed/<company>/<doc_id>/vector_store/`: the actual vector DB files (SQLite/Chroma/FAISS/Qdrant dump) for retrieval.
- `data/evals/per_doc/<doc_id>/`: reference QA sets and manual labels for per-document evaluation (accuracy, citation relevance).
- `data/evals/cross_company/`: question sets that touch multiple companies (e.g., revenue/margin comparisons).
- `data/staging/`: scratch space for conversions and temporary artifacts; safe to clean.
- `pipelines/`: code for ingestion/chunking/embedding pipelines (e.g., DOCX extraction and chunk generation).

### Demo
- Run `python -m pipelines.demo_chunking` to see paragraph-level blocks and how they form overlapping chunks (`chunk_size_tokens=12`, `overlap_tokens=4`).

## Tooling / Setup
- PDM project (`pyproject.toml`) with `python-docx` for DOCX extraction. Requires Python >= 3.11.
- Standard setup:
  1) Install PDM (see https://pdm.fming.dev/latest/#installation).
  2) Point PDM at your interpreter: `pdm use -f $(which python3.12)` (or any >=3.11).
  3) Install deps: `pdm install`.
  4) Run modules via PDM: `pdm run python -m pipelines.demo_chunking`.

## Workflow (per document)
1) Register the source in `data/index/documents.json` with a `doc_id`, `source_file`, type (filing vs call), format, `questions_file`, optional `eval_reference_file`, and processing paths.  
2) Extract text to `.../text/` (keep page/section numbers).  
3) Chunk text to `.../chunks/chunks.jsonl`, preserving `chunk_id` and metadata to enable traceable citations.  
4) Embed chunks to `.../embeddings/` (Parquet/JSONL) with `chunk_id`, `text`, `embedding`, and metadata.  
5) Persist embeddings into `.../vector_store/` while maintaining a mapping back to `chunk_id`/text so every vector is explainable.  
6) Keep question sets in `data/questions/...` keyed by `doc_id`; store reference answers/manual labels in `data/evals/...` for the evaluation dashboard.  
7) For cross-company questions, pull context from multiple `doc_id` entries and log results under `data/evals/cross_company/`.

## Conventions
- Use lowercase, snake_case `doc_id` strings that include company + year + doc type (`microsoft_2024_annual_report`, `microsoft_fy24_q4_10k`, `microsoft_2024_q4_earnings_call`).
- Reuse the same `doc_id` in the manifest, processed directories, question/eval filenames, and any configs.
- For citations, include `page_span` or `section` metadata in chunks and propagate that into embeddings/vector store entries.
- Keep embeddings bundled with text/metadata so each vector can be resolved to readable context in the UI/source viewer.
