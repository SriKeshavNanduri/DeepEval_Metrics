# # Import necessary libraries
import ast
from openai import OpenAI
import os 
from dotenv import load_dotenv

# load environment variables
load_dotenv()
# OpenAI client initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read the saved .lst file back into a Python list
with open('retrieved_results.lst', "r", encoding="utf-8") as f:
    retrieved_results = ast.literal_eval(f.read())
    
print("Retrieved Results ------------------------------")
for i, line in enumerate(retrieved_results):
    
    print(f"{i + 1}, {line}")
print("------------------------------")

query = "when did Steve jobs associated with Pixar studio?"
llm_answer = "Steve Jobs associated with Pixar studio in 1986 when he purchased the company and became its chairman and majority shareholder."
##################################################################################################

# Metrics and Evaluation
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig


from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric, 
    ContextualRecallMetric, 
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

print("LLM Answer ---> " ,llm_answer)
ground_truth = "Steve Jobs was associated with Pixar Studio from 1986."
print("Ground Truth ---> " ,ground_truth)

# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------

"""
AnswerRelevancyMetric --> Measures how relevant the LLM's answer is to the question, based on the retrieved context. 
IMPORTANT : 
It Does NOT check for the correctness of the answer, but rather its relevance to the question and the context.

Example 1 : if my llm_answer is "Steve Jobs is not associated with Pixar Studio." then the answer relevancy score will be low because the answer is not relevant to the question and the retrieved context.
"""
llm_answer_test_answer_relevancy_1 = "neil armstrong is the first man to walk on the moon."
llm_answer_test_answer_relevancy_2 = "Steve jobs associated with Pixar Studio in 1999."

print("llm_answer_test_answer_relevancy_1", llm_answer_test_answer_relevancy_1)
print("llm_answer_test_answer_relevancy_2", llm_answer_test_answer_relevancy_2)

answer_relevancy_metric = [
    AnswerRelevancyMetric(
        model="gpt-3.5-turbo",
        include_reason=True
    ) ] 

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer_test_answer_relevancy_2, # llm_answer_test_answer_relevancy
    expected_output=ground_truth,
    retrieval_context=retrieved_results
)

# check on what metric you are looking into among the 5 Deepeval metrics.
metric = answer_relevancy_metric

# # # ----------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------

"""
FaithfulnessMetric --> Evaluates whether the actual_output factually aligns with the contents of your retrieval_context 
IMPORTANT : 
It checks for the correctness of the answer that is reterieved from the retrieved context. It does not check for the relevance of the answer to the question.

Example 1 : if my llm_answer is "Steve Jobs is associated with Pixar Studio and steve jobs worked for apple till 2007." then the answer relevancy score will be low because the answer is not relevant to the question and the retrieved context.
"""

llm_answer_test_faithfulness = 'Steve Jobs was not associated with Pixar Studio until 2007' 
print("llm_answer_test_faithfulness:", llm_answer_test_faithfulness)

faithfulness_metric = [
    FaithfulnessMetric(
        model="gpt-3.5-turbo",
        include_reason=True
    )] 

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer_test_faithfulness, #llm_answer_test_faithfulness
    expected_output=ground_truth,
    retrieval_context=retrieved_results
)

# check on what metric you are looking into among the 5 Deepeval metrics.
metric = faithfulness_metric


# # # ----------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------

"""
ContextualPrecisionMetric --> Measures the precision of the LLM's answer based on the retrieved context. 
Among the chunks retrieved, are the most relevant chunks appearing at the top rather than lower down?
"""

retrieved_results_contextual_precision = retrieved_results[2:3] # the least relevant chunk among the retrieved results based on the distance returned by FAISS.
retrieved_results_contextual_precision_2 = retrieved_results[2:3]*10
retrieved_results_contextual_precision_2.append(retrieved_results[0]) 
print("retrieved_results_contextual_precision:", retrieved_results_contextual_precision)

contextual_precision_metrics = [
    ContextualPrecisionMetric(
        model="gpt-3.5-turbo",
        include_reason=True
    )]

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer, 
    expected_output=ground_truth,
    retrieval_context=retrieved_results_contextual_precision_2 # retrieved_results_contextual_precision
)

# check on what metric you are looking into among the 5 Deepeval metrics.
metric = contextual_precision_metrics
# # # ----------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------

"""
ContextualRecallMetric --> measure the quality of your RAG pipeline's retriever by evaluating the extent of which the retrieval_context aligns with the ground_truth answer. 
IMPORTANT : If my contextual recall score is close to 1 means we are getting all the necessary chunks that contain the answer to the question, if my contextual recall score is close to 0 means we are not getting all the necessary chunks that contain the answer to the question.
"""
retrieved_results_contextual_recall =["Company in 1976. After the company's board of directors fired him in 1985", "steve becoming Pixar chairman and majority shareholder until 2007. Jobs returned to Apple in 1997 as CEO" ] 

print("retrieved_results_contextual_recall:", retrieved_results_contextual_recall)

contextual_recall_metrics = [
    ContextualRecallMetric(
        model="gpt-3.5-turbo",
        include_reason=True
    )]

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer, #llm_answer
    expected_output=ground_truth,
    retrieval_context=retrieved_results # retrieved_results_contextual_precision
)

# check on what metric you are looking into among the 5 Deepeval metrics.
metric = contextual_recall_metrics

# # # -----------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------

'''
ContextualRelevancyMetric --> Measures the relevance of the retrieved context to the question and the LLM's answer.
IMPORTANT : If this metric is close to 1 means we are retrieving the most relevant chunks that contain the answer to the question, if this metric is close to 0 means we are not retrieving the relevant chunks that contain the answer to the question.
'''

contextual_relevancy_metrics = [
    ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-3.5-turbo",
        include_reason=True
    )
]
test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer, #llm_answer
    expected_output=ground_truth,
    retrieval_context=retrieved_results
)

metric = contextual_relevancy_metrics
# # # -----------------------------------------------------------------------------\
# # # -----------------------------------------------------------------------------

# # -----------------------------
# # Run Evaluation
# # -----------------------------

from contextlib import redirect_stdout, redirect_stderr
import io

f = io.StringIO()

with redirect_stdout(f), redirect_stderr(f):
    results = evaluate(
        test_cases=[test_case],
        metrics=metric
    )

metric[0].measure(test_case)

print("\n====================")
print("Metric:", metric[0].__class__.__name__)
print("Score :", metric[0].score)
print("Reason:", metric[0].reason)