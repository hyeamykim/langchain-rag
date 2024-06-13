from query_data import query_rag
from get_embedding_function import get_embedding_function

from langchain_openai import OpenAI
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_sustainability_docs1():
    assert query_and_validate(
        question="Where did COP26 take place?",
        expected_response="It took place in Glasgow",
    )
def test_sustainability_docs2():
    assert query_and_validate(
        question="Who are the environmental activists mentioned in the report?",
        expected_response="David Attenborouch, Greta Thunberg",
    )
def test_sustainability_dos3():
    assert query_and_validate(
        question="What are the barriers Uk businesses mentioned to taking sustainable actions?",
        expected_response="organizational accountability, mismanagement of data, lack of understanding",
    )
def test_sustainability_docs4():
    assert query_and_validate(
        question="How many respondents participated in the SAP survey in 2022? From how many industries?",
        expected_response="328 respondents from 29 industries",
    )
def test_sustainability_docs5():
    assert query_and_validate(
        question="What are the top 5 barriers to taking sustainability action according to the SAP sustainability report 2023?",
        expected_response="lack of funding, difficulty proving the ROI of sustainable actions, lack of support from senior stakeholders",
    )
def test_sustainability_docs6():
    assert query_and_validate(
        question="Which environmental issues are UK businesses directly measuring?",
        expected_response="material use, resource availability, fresh water availability",
    )
def test_sustainability_docs7():
    assert query_and_validate(
        question="How much did VCs invest in carbon accounting in 2022?",
        expected_response="$767 million",
    )
def test_sustainability_docs8():
    assert query_and_validate(
        question="What are the 3 scopes of industry's emissions? What does each scope mean?",
        expected_response="scope 1 is direct emissions by company, scope 2 is indirect emissions such as energy purchase, and scope 3 is third-party emissions from supply chain for example ",
    )
def test_sustainability_docs9():
    assert query_and_validate(
        question="When is UK's EPR scheme going to be introduced?",
        expected_response="2025",
    )
def test_sustainability_docs10():
    assert query_and_validate(
        question="Who's held responsible for setting the strategic direction on sustainability action?",
        expected_response="board of directors, CEOs, Chief Sustainability Officers, COOs, CFOs",
    )
def test_sustainability_docs11():
    assert query_and_validate(
        question="What kind of SAP products can help with UK's EPR scheme?",
        expected_response="SAP Responsible Design and Production, SAP Sustainability Control Tower, SAP Sustainability Footprint Management",
    )
def test_sustainability_docs12():
    assert query_and_validate(
        question="What are SAP's sustainability goals?",
        expected_response="having their top one hundred suppliers reporting company-wide and product-level emissions "
                          "for key products by 2027, achieving net-zero emissions across our value chain by 2030",
    )
def test_sustainability_docs13():
    assert query_and_validate(
        question="Is there a standardized approach towards sustainability reporting?",
        expected_response="no",
    )
def test_sustainability_docs14():
    assert query_and_validate(
        question="what are some ESG goals of companies by 2030?",
        expected_response="Unilever is going to provide living wage to everyone in its value chain by 2030. "
                          "Norwegian Air wants to reduce carbon emission by 45% by 2030",
    )
def test_sustainability_docs15():
    assert query_and_validate(
        question="How did SAP help other companies' sustainable actions?",
        expected_response="SAP Sustainability Control Tower helped msg global publish its first sustainability report in 2022,",
    )



def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OpenAI()
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    evaluator = load_evaluator("embedding_distance", embeddings=get_embedding_function(), distance_metric=EmbeddingDistance.EUCLIDEAN)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # get the pairwise distance between the expected result and actual result embeddings
        prob_result = evaluator.evaluate_strings(
            prediction=response_text, reference=expected_response
        )
        if prob_result['score'] >= 0.8:
            print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return True
        else:
            # Print response in Red if it is incorrect.
            print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


