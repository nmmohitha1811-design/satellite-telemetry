from graph_pipeline import build_graph

pipeline = build_graph()

# Manually inject a channel_id, but we'll also monkey-test the routing
# by checking what happens with a channel unlikely to match our corpus well.
# D-8 and A-9 were real misses in your Layer 1 results (no class info helps retrieval).

result = pipeline.invoke({"channel_id": "D-8"})

print(f"Channel: {result['channel_id']} ({result['spacecraft']})")
print(f"Retrieval score: {result['best_retrieval_score']:.3f}")
print(f"Grounded: {result['is_grounded']}")
print(f"\n--- FINAL REPORT ---\n{result['final_report']}")