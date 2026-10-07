from rag_answer import retrieve

results = retrieve("what are the satellites info do you have and what is the sole mission of these satellites")

for r in results:
    print(f"{r['score']:.3f} - {r['source']} chunk {r['chunk_index']}")
    print(r['text'][:150])
    print()