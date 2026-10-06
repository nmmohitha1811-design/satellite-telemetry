import json
import numpy as np
from sentence_transformers import SentenceTransformer

RAG_PATH = r"C:\Users\ADMIN\space-copilot\rag-copilot"

print("Loading embedding model and index...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

with open(f"{RAG_PATH}/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(f"{RAG_PATH}/chunk_embeddings.npy")


def build_query(detector_state):
    """Turn the Detector agent's findings into a natural-language search query."""
    spacecraft_name = "Curiosity rover" if detector_state["spacecraft"] == "MSL" else "SMAP satellite"
    query = (
        f"What causes a {detector_state['anomaly_class']} anomaly or fault "
        f"on the {spacecraft_name}?"
    )
    return query


def retrieve(detector_state, top_k=3):
    query = build_query(detector_state)

    query_vec = embed_model.encode([query])[0]
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

    state_update = {
        "retrieved_chunks": results,
        "best_retrieval_score": results[0]["score"] if results else 0.0,
    }
    return query, state_update


if __name__ == "__main__":
    from detector_agent import detect_anomaly

    detector_state = detect_anomaly()
    print("Detector output:")
    print(json.dumps(detector_state, indent=2))

    query, retrieval_state = retrieve(detector_state)
    print(f"\nGenerated query: \"{query}\"")
    print(f"\nBest retrieval score: {retrieval_state['best_retrieval_score']:.3f}")
    for i, chunk in enumerate(retrieval_state["retrieved_chunks"]):
        print(f"\n--- Result {i+1} (score {chunk['score']:.3f}) ---")
        print(f"Source: {chunk['source']} (chunk {chunk['chunk_index']})")
        print(chunk['text'][:200] + "...")