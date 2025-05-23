from sentence_transformers import SentenceTransformer
from t5_inference import get_response
import faiss
import numpy as np
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')
loaded_index = faiss.read_index("faiss_index.bin")

df = pd.read_csv('Dataset/A_buffett_qa.csv')

def generate_answer_t5(query):
    return get_response(query)[0]

def answer_with_faiss_or_t5(query,index=loaded_index, k=1, threshold=0.8):
    # query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # search for top K
    distances, indices = index.search(query_embedding, k)

    # extract answer
    results = []
    t5_answer = ""
    for idx, score in zip(indices[0], distances[0]):
        if score >= threshold:
            results.append(df.iloc[idx]["answer"])
        else:
            t5_answer = generate_answer_t5(query)
            results.append(t5_answer)
    return results[0]








