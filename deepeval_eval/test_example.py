from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel

# def test_correctness():

#     model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")

#     correctness_metric = GEval(
#         model=model,
#         name="Correctness",
#         criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
#         evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
#         threshold=0.5
#     )
#     test_case = LLMTestCase(
#         model=model,
#         input="I have a persistent cough and fever. Should I be worried?",
#         # Replace this with the actual output from your LLM application
#         actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
#         expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
#     )
#     assert_test(test_case, [correctness_metric])




def test_case():

    # model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")

    correctness_metric = GEval(
        # model=model,
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        # model=model,
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    assert_test(test_case, [correctness_metric])