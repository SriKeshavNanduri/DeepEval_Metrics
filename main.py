# Import necessary libraries
import faiss
import numpy as np
from openai import OpenAI
import os 
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# OpenAI client initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chunking the text data
def chunk_text(file_path, chunk_size=300, overlap=30):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

text_chunks = chunk_text("rag_data.txt")

# 2. CREATE METADATA
# In this case, we map the list index to the chunk content
metadata = {i: {"text": chunk, "source": "rag_data.txt"} for i, chunk in enumerate(text_chunks)}

# 3. GENERATE EMBEDDINGS
def get_embeddings(text_list):
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    # Convert to float32 (FAISS requirement)
    return np.array([res.embedding for res in response.data]).astype('float32')

embeddings = get_embeddings(text_chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Total vectors in index: {index.ntotal}")

# 4. QUERYING
# Convert query to vector
query = "when did Steve jobs associated with Pixar studio?"
query_vector = get_embeddings([query])

# Search FAISS (k=3 means get top 3 results)
distances, indices = index.search(query_vector, k=3)

# Retrieve actual text using the indices
retrieved_results = []
for idx in indices[0]:
    retrieved_results.append(metadata[idx]['text'])


print("-------------------------------")
print("Query by User: ", query)
print("-------------------------------")
print("retrieved_results:")
for i, info in enumerate(retrieved_results, start=1):
    print(f"{i}, {info}")
print("-------------------------------")

response = client.chat.completions.create(
        model="gpt-3.5-turbo", # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant. you will answer the question based on the retrieved context along with the metadata. If the context does not contain the answer, say you don't know."},
            {"role": "user", "content": "Context: " + metadata[indices[0][0]]['text'] + "\n\nQuestion: " + query}
            
        ]
    )

##################################################################################################33

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



llm_answer = response.choices[0].message.content
print("LLM Answer ---> " ,llm_answer)
ground_truth = "Steve Jobs was associated with Pixar Studio from 1986."
print("Ground Truth ---> " ,ground_truth)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

"""
AnswerRelevancyMetric --> Measures how relevant the LLM's answer is to the question, based on the retrieved context. 
IMPORTANT : 
It Does NOT check for the correctness of the answer, but rather its relevance to the question and the context.

Example 1 : if my llm_answer is "Steve Jobs is not associated with Pixar Studio." then the answer relevancy score will be low because the answer is not relevant to the question and the retrieved context.
"""
llm_answer_test_answer_relevancy = "neil armstrong is the first man to walk on the moon."

print("llm_answer_test_answer_relevancy", llm_answer_test_answer_relevancy)

answer_relevancy_metric = [
    AnswerRelevancyMetric(
        model="gpt-4o-mini",
        include_reason=True
    ) ] 

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer_test_answer_relevancy, # llm_answer_test_answer_relevancy
    expected_output=ground_truth,
    retrieval_context=retrieved_results
)

# check on what metric you are looking into among the 5 Deepeval metrics.
metric = answer_relevancy_metric

# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------

# """
# FaithfulnessMetric --> Evaluates whether the actual_output factually aligns with the contents of your retrieval_context 
# IMPORTANT : 
# It checks for the correctness of the answer that is reterieved from the retrieved context. It does not check for the relevance of the answer to the question.

# Example 1 : if my llm_answer is "Steve Jobs is associated with Pixar Studio and steve jobs worked for apple till 2007." then the answer relevancy score will be low because the answer is not relevant to the question and the retrieved context.
# """

# llm_answer_test_faithfulness = 'Steve Jobs was not associated with Pixar Studio until 2007' 
# print("llm_answer_test_faithfulness:", llm_answer_test_faithfulness)

# faithfulness_metric = [
#     FaithfulnessMetric(
#         model="gpt-4o-mini",
#         include_reason=True
#     )] 

# test_case = LLMTestCase(
#     input=query,
#     actual_output=llm_answer, #llm_answer_test_faithfulness
#     expected_output=ground_truth,
#     retrieval_context=retrieved_results
# )

# # check on what metric you are looking into among the 5 Deepeval metrics.
# metric = faithfulness_metric


# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------

# """
# ContextualPrecisionMetric --> Measures the precision of the LLM's answer based on the retrieved context. 
# Among the chunks retrieved, are the most relevant chunks appearing at the top rather than lower down?
# """

# retrieved_results_contextual_precision = retrieved_results[2:3] # the least relevant chunk among the retrieved results based on the distance returned by FAISS.
# print("retrieved_results_contextual_precision:", retrieved_results_contextual_precision)

# contextual_precision_metrics = [
#     ContextualPrecisionMetric(
#         model="gpt-4o-mini",
#         include_reason=True
#     )]

# test_case = LLMTestCase(
#     input=query,
#     actual_output=llm_answer, 
#     expected_output=ground_truth,
#     retrieval_context=retrieved_results # retrieved_results_contextual_precision
# )

# # check on what metric you are looking into among the 5 Deepeval metrics.
# metric = contextual_precision_metrics
# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------

# """
# ContextualRecallMetric --> measure the quality of your RAG pipeline's retriever by evaluating the extent of which the retrieval_context aligns with the ground_truth answer. 
# IMPORTANT : If my contextual recall score is close to 1 means we are getting all the necessary chunks that contain the answer to the question, if my contextual recall score is close to 0 means we are not getting all the necessary chunks that contain the answer to the question.
# """
# retrieved_results_contextual_recall =["Company in 1976. After the company's board of directors fired him in 1985", "steve becoming Pixar chairman and majority shareholder until 2007. Jobs returned to Apple in 1997 as CEO" ] 

# print("retrieved_results_contextual_recall:", retrieved_results_contextual_recall)

# contextual_recall_metrics = [
#     ContextualRecallMetric(
#         model="gpt-4o-mini",
#         include_reason=True
#     )]

# test_case = LLMTestCase(
#     input=query,
#     actual_output=llm_answer, #llm_answer
#     expected_output=ground_truth,
#     retrieval_context=retrieved_results # retrieved_results_contextual_precision
# )

# # check on what metric you are looking into among the 5 Deepeval metrics.
# metric = contextual_recall_metrics

# # -----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------

# '''
# ContextualRelevancyMetric --> Measures the relevance of the retrieved context to the question and the LLM's answer.
# IMPORTANT : If this metric is close to 1 means we are retrieving the most relevant chunks that contain the answer to the question, if this metric is close to 0 means we are not retrieving the relevant chunks that contain the answer to the question.
# '''

# contextual_relevancy_metrics = [
#     ContextualRelevancyMetric(
#         threshold=0.7,
#         model="gpt-4o-mini",
#         include_reason=True
#     )
# ]
# test_case = LLMTestCase(
#     input=query,
#     actual_output=llm_answer, #llm_answer
#     expected_output=ground_truth,
#     retrieval_context=retrieved_results
# )

# metric = contextual_relevancy_metrics
# # -----------------------------------------------------------------------------\
# # -----------------------------------------------------------------------------

# -----------------------------
# Run Evaluation
# -----------------------------

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