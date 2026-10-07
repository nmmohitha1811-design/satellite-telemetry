import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

SIMILARITY_THRESHOLD = 0.35  # below this, we don't trust the match enough to answer

print("Loading embedding model and index...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

RAG_COPILOT_DIR = r"C:\Users\ADMIN\space-copilot\rag-copilot"

with open(f"{RAG_COPILOT_DIR}/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(f"{RAG_COPILOT_DIR}/chunk_embeddings.npy")


def retrieve(query, top_k=3):
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
    return results


def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[Source {i+1}: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a technical assistant. Answer the question using ONLY the information in the sources below. 
Do not use any outside knowledge. If the sources do not contain enough information to answer, say exactly: 
"I don't have enough grounded information to answer that."

When you use information from a source, cite it in brackets like [Source 1] or [Source 2].

SOURCES:
{context_text}

QUESTION: {query}

ANSWER:"""
    return prompt


def answer_question(query, top_k=3):
    retrieved = retrieve(query, top_k=top_k)
    best_score = retrieved[0]["score"]

    print(f"\n[Retrieval] Best match score: {best_score:.3f}")
    for i, r in enumerate(retrieved):
        print(f"  Source {i+1}: {r['source']} (chunk {r['chunk_index']}, score {r['score']:.3f})")

    if best_score < SIMILARITY_THRESHOLD:
        print("\n[Guardrail triggered: best match score below threshold]")
        return "I don't have enough grounded information to answer that."

    prompt = build_prompt(query, retrieved)

    print("\n[Generating answer with local LLM...]")
    response = ollama.chat(model='llama3.2-local', messages=[
        {"role": "user", "content": prompt}
    ])

    return response['message']['content']


if __name__ == "__main__":
    while True:
        query = input("\n\nEnter a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        answer = answer_question(query)
        print(f"\n--- ANSWER ---\n{answer}")