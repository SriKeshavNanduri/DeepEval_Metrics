import faiss
import numpy as np
from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    print(f"Retrieved Context idx {idx} : {metadata[idx]['text']}")
    print("-------------------------------")
    retrieved_results.append(metadata[idx]['text'])

response = client.chat.completions.create(
        model="gpt-3.5-turbo", # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant. you will answer the question based on the retrieved context along with the metadata. If the context does not contain the answer, say you don't know."},
            {"role": "user", "content": "Context: " + metadata[indices[0][0]]['text'] + "\n\nQuestion: " + query}
            
        ]
    )
# Print the result
print("Answer:-----------------------------------------")
llm_answer = response.choices[0].message.content
print(llm_answer)
ground_truth = "Steve Jobs was associated with Pixar Studio from 1986 to 2006, later it was acquired by The Walt Disney Company."


from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=query,
    actual_output=llm_answer,
    expected_output=ground_truth,
    retrieval_context=retrieved_results
)


metrics = [
    AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    ),

    FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    ),

    ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    )
]


# -----------------------------
# Run Evaluation
# -----------------------------

evaluate(
    test_cases=[test_case],
    metrics=metrics,
    show_indicator=False,
    print_results=False
)


for metric in metrics:
    metric.measure(test_case)

    print("\n====================")
    print("Metric:", metric.__class__.__name__)
    print("Score :", metric.score)
    print("Reason:", metric.reason)