import os
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.config.settings import get_settings


dataset = EvaluationDataset(goldens=[Golden(input="What's the weather like today?")])
model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "360")

settings = get_settings()
print(f"Using DeepEval settings: {settings.dict()}")

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=model.generate(golden.input)[0]
    )
    dataset.add_test_case(test_case)

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=model)
    assert_test(test_case, [answer_relevancy_metric])
