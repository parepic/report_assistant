from __future__ import annotations

import os

from deepeval import evaluate
from deepeval.config.settings import get_settings
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from ollama_eval_model import OllamaEvalModel
from deepeval.models import OllamaModel


def main() -> None:
    # Increase per-attempt timeout for slow local models (seconds).
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")

    settings = get_settings()
    print(f"Using DeepEval settings: {settings.dict()}")

    # --- Curated, no-external-docs setup ---
    question = "What is the capital of France?"
    expected_answer = "Paris."

    # Pretend this is your retrieved context.
    # Try swapping in irrelevant text to see retrieval metrics drop.
    retrieved_context = [
        "France is a country in Western Europe. Its capital city is Paris.",
        "Paris is known for the Eiffel Tower.",
    ]

    # Start manual: hardcode a candidate answer.
    # Later you can swap this with your real RAG answer.
    actual_answer = "The capital of France is Paris."

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieved_context,
    )

    # Ollama-backed evaluator (used for scoring)
    # Toggle between standard DeepEval OllamaModel and the custom eval model.
    use_custom_eval_model = False
    if use_custom_eval_model:
        eval_llm = OllamaEvalModel(model="llama3.1:8b", base_url="http://localhost:11434")
    else:
        eval_llm = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")

    retrieval_metrics = [
        # Keep the smallest set first: one retrieval metric
        ContextualPrecisionMetric(model=eval_llm),
    ]

    response_metrics = [
        # Keep the smallest set first: one response metric
        AnswerRelevancyMetric(model=eval_llm),
    ]

    print("Retrieval metrics:")
    evaluate([test_case], retrieval_metrics)

    print("\nResponse metrics:")
    evaluate([test_case], response_metrics)


if __name__ == "__main__":
    main()
