"""DeepEval RAG evaluation runner.

This script:
1) Loads a small question set.
2) Runs the same RAG pipeline used in the app (retrieve -> prompt -> answer).
3) Builds DeepEval test cases with retrieval context attached.
4) Evaluates retrieval quality and answer quality with LLM-as-judge metrics.
"""

import json
import os
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# Ensure repo root is on sys.path so sibling packages are importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from report_assistant.data_classes import compute_strategy_hash
from report_assistant.llm import llm_generate, retrieve_top_k_from_qdrant
from report_assistant.utils.load_utils import load_global_config


# Load questions from a JSON file.
QUESTIONS_PATH = Path("data/questions/Amazon/amazon_10-k-item1a.json")
COLLECTION_NAME = "company__amazon"

raw = QUESTIONS_PATH.read_text(encoding="utf-8")
items = json.loads(raw)
items = items[:2]

# Load global config and compute strategy hash.
config = load_global_config()
strategy_hash = compute_strategy_hash(config.chunk_strategy)

ollama_url = config.OLLAMA_URL
qdrant_url = config.QDRANT_URL
llm_model = config.LLM_MODEL
embed_model = config.chunk_strategy.embed_model
top_k = config.top_k



# Set up DeepEval evaluation.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "360")
model = OllamaModel(model=llm_model, base_url=ollama_url)

goldens = [
    Golden(input=item["question"], expected_output=item["expected_answer"])
    for item in items
]
dataset = EvaluationDataset(goldens=goldens)

print(f"Loaded {len(dataset.goldens)} goldens from {QUESTIONS_PATH}")

# Generate test cases by retrieving context and getting model responses.
for golden in dataset.goldens:
    retrieved_chunks = retrieve_top_k_from_qdrant(
        golden.input,
        COLLECTION_NAME,
        qdrant_url,
        ollama_url,
        embed_model,
        strategy_hash=strategy_hash,
        k=top_k,
    )
    # Build the same prompt format used in the app.
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"Chunk {i + 1}:\n{chunk}\n\n"
    prompt = (
        "You are a helpful assistant answering questions about a company document.\n"
        "Use ONLY the information in the context below. If the answer is not there,\n"
        "say you don't know and do not make things up.\n\n"
        f"Context:\n{context}\n"
        f"Question: {golden.input}\n\n"
        "Answer:\n"
    )
    print(f"Generating response for question: {golden.input}")
    response_text = llm_generate(prompt, ollama_url, llm_model)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response_text,
        expected_output=golden.expected_output,
        retrieval_context=retrieved_chunks,
    )
    dataset.add_test_case(test_case)

# Evaluate retrieval metrics (context quality).
retrieval_metrics = [
    # Keep the smallest set first: one retrieval metric
    ContextualPrecisionMetric(model=model),
]

# Evaluate response metrics (answer quality).
response_metrics = [
    # Keep the smallest set first: one response metric
    AnswerRelevancyMetric(model=model),
]

print("Evaluating retrieval metrics:")
evaluate(dataset.test_cases, retrieval_metrics)
print("Evaluating response metrics:")
evaluate(dataset.test_cases, response_metrics)
