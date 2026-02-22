from __future__ import annotations

from deepeval import evaluate
from deepeval.config.settings import get_settings
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from ollama_eval_model import OllamaEvalModel
from deepeval.models import OllamaModel

from app.utils.load_utils import load_global_config


# Manual evaluation setup
# Use this script to get an understanding of how DeepEval works with a simple, hardcoded example.
def main() -> None:
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
    use_ollama = False
    if use_custom_eval_model:
        eval_llm = OllamaEvalModel(model="llama3.1:8b", base_url="http://localhost:11434")
    elif use_ollama:
        eval_llm = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
    else:
        # Use openai by default
        config = load_global_config()
        print(config)
        eval_llm = config.get("LLM_MODEL_EVAL", "gpt-4o-mini")


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
