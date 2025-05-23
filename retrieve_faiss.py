from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

loaded_index = faiss.read_index("faiss_index.bin")

df = pd.read_csv('Dataset/A_buffett_qa.csv')

def search_answer(query,index=loaded_index, k=1, threshold=0.7):
    # query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # search for top K
    distances, indices = index.search(query_embedding, k)

    # extract answer
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if score >= threshold:
            results.append(df.iloc[idx]["answer"])
        else:
            results.append('')
    return results[0]

