import json
import ollama


def build_report_prompt(detector_state, retrieved_chunks):
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[Source {i+1}: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a mission-operations assistant drafting a short incident note.

DETECTED ANOMALY:
{detector_state['detection_summary']}

RELEVANT SOURCES:
{context_text}

Write a brief incident report (3-5 sentences) that:
1. States what was detected (use the detection summary above).
2. Explains, using ONLY the sources above, what kind of cause or precedent this resembles.
3. Cites sources in brackets like [Source 1].
4. If the sources do not clearly explain a specific cause, say so honestly rather than guessing.

INCIDENT REPORT:"""
    return prompt


def write_report(detector_state, retrieved_chunks):
    prompt = build_report_prompt(detector_state, retrieved_chunks)
    response = ollama.chat(model='llama3.2-local', messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']


if __name__ == "__main__":
    from detector_agent import detect_anomaly
    from retrieval_agent import retrieve

    detector_state = detect_anomaly()
    query, retrieval_state = retrieve(detector_state)

    print("Detection:", detector_state["detection_summary"])
    print(f"\nRetrieval score: {retrieval_state['best_retrieval_score']:.3f}")
    print("\nGenerating report...\n")

    report = write_report(detector_state, retrieval_state["retrieved_chunks"])
    print("--- DRAFT REPORT ---")
    print(report)