import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("chunk_embeddings.npy")

def search(query, top_k=3):
    query_vec = model.encode([query])[0]

    # cosine similarity between the query and every chunk
    dot_products = np.dot(embeddings, query_vec)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    similarities = dot_products / norms

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            "source": chunks[idx]["source"],
            "chunk_index": chunks[idx]["chunk_index"],
            "text": chunks[idx]["text"]
        })
    return results

if __name__ == "__main__":
    while True:
        query = input("\nEnter a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        results = search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (score: {r['score']:.3f}) ---")
            print(f"Source: {r['source']} (chunk {r['chunk_index']})")
            print(r['text'][:300] + "...")